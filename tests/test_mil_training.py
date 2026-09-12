"""Actual CPU fitting, validation selection, metrics and resumable checkpoints."""

import copy
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from lightning.pytorch.utilities.exceptions import SIGTERMException  # noqa: E402

from histopilot.training.fold import train_fold  # noqa: E402
from histopilot.training.module import (  # noqa: E402
    MILTrainModule,
    classification_metrics,
    prediction_rows,
)


def target(classes=2, unit="patient"):
    return {
        "task": "binary_classification" if classes == 2 else "multiclass_classification",
        "unit": unit,
        "classes": [f"class-{index}" for index in range(classes)],
        "positiveClass": "class-0" if classes == 2 else None,
    }


def tiny_plan(tmp_path, classes=2):
    records, files = [], {}
    rng = np.random.default_rng(13)
    for role in ("train", "val", "test"):
        for label in range(classes):
            for repetition in range(2):
                identity = f"{role}-{label}-{repetition}"
                path = tmp_path / f"{identity}.npy"
                values = rng.normal(0, 0.1, (5 + repetition, 4)).astype(np.float32)
                values[:, label] += 1
                np.save(path, values)
                files[identity] = {
                    "slideId": identity,
                    "path": str(path),
                    "format": "npy",
                    "dimensions": 4,
                    "patchCount": len(values),
                }
                records.append(
                    {
                        "slideId": identity,
                        "patientId": identity,
                        "label": f"class-{label}",
                        "partition": role,
                        "pool": "development",
                        "phase": "evaluation",
                    }
                )
    return {
        "runId": "run-tiny",
        "trainingSeed": 42,
        "target": target(classes),
        "device": "cpu",
        "resources": {"cpuThreadsPerRun": 1, "dataLoaderWorkers": 0},
        "recipe": {
            "model": "abmil",
            "learningRate": 0.02,
            "weightDecay": 0,
            "optimizer": "adamw",
            "maxEpochs": 2,
            "batchSize": 2,
            "bagSize": 3,
            "earlyStopping": False,
            "patience": 5,
            "checkpointMetric": "validation_loss",
            "embedDim": 8,
            "attentionDim": 4,
            "numFcLayers": 1,
            "gatedAttention": True,
            "dropout": 0,
            "inputDropout": 0,
            "gradientCheckpointing": False,
        },
        "data": {
            "memberships": records,
            "featureFiles": files,
            "loadingPolicy": "native",
            "packPath": None,
            "featureDim": 4,
        },
    }


def test_patient_metrics_weight_patients_once_and_honor_positive_class_order():
    rows = [
        {
            "slideId": "a",
            "patientId": "a",
            "labelIndex": 0,
            "label": "class-0",
            "probabilities": [0.9, 0.1],
        },
        *[
            {
                "slideId": f"b-{index}",
                "patientId": "b",
                "labelIndex": 1,
                "label": "class-1",
                "probabilities": [0.8, 0.2],
            }
            for index in range(3)
        ],
    ]
    metrics = classification_metrics(rows, target())
    assert metrics["slide"]["accuracy"] == 0.25
    assert metrics["patient"]["accuracy"] == 0.5
    assert metrics["selected"] == metrics["patient"]
    assert metrics["patient"]["auroc"] == 1
    assert metrics["patient"]["loss"] == pytest.approx((-np.log(0.9) - np.log(0.2)) / 2)
    assert classification_metrics(rows, target(unit="slide"))["selected"] == metrics["slide"]


def test_missing_assessment_class_reports_undefined_auc_instead_of_zero():
    rows = [
        {
            "slideId": "a",
            "patientId": "a",
            "labelIndex": 0,
            "label": "class-0",
            "probabilities": [0.9, 0.1],
        }
    ]
    summary = classification_metrics(rows, target())
    assert summary["selected"]["auroc"] is None
    assert summary["selected"]["missingClasses"] == ["class-1"]


@pytest.mark.parametrize(
    "classes,probabilities,expected_f1,expected_balanced_accuracy",
    [
        (2, [[0.9, 0.1], [0.8, 0.2]], 0.5, 1.0),
        (2, [[0.9, 0.1], [0.1, 0.9]], 1 / 3, 0.5),
        (3, [[0.9, 0.05, 0.05], [0.05, 0.05, 0.9]], 2 / 9, 0.5),
    ],
)
def test_macro_f1_preserves_frozen_class_average_with_missing_and_predicted_only_classes(
    classes, probabilities, expected_f1, expected_balanced_accuracy
):
    rows = [
        {
            "slideId": f"slide-{index}",
            "patientId": f"patient-{index}",
            "labelIndex": 0,
            "label": "class-0",
            "probabilities": values,
        }
        for index, values in enumerate(probabilities)
    ]
    metrics = classification_metrics(rows, target(classes))
    for unit in ("slide", "patient"):
        # Existing reports average every frozen class with zero for undefined
        # F1; predicted-only classes also have a defined F1 of zero.
        assert metrics[unit]["macroF1"] == pytest.approx(expected_f1)
        assert metrics[unit]["balancedAccuracy"] == expected_balanced_accuracy
        assert metrics[unit]["missingClasses"] == target(classes)["classes"][1:]


def test_extreme_finite_logits_retain_true_loss_for_checkpoint_comparison():
    rows = prediction_rows(
        {"labels": torch.tensor([1]), "slideIds": ["a"], "patientIds": ["a"]},
        torch.tensor([[1000.0, 0.0]]),
        target(),
    )
    assert rows[0]["probabilities"] == [1.0, 0.0]
    assert classification_metrics(rows, target())["selected"]["loss"] == 1000


@pytest.mark.parametrize(
    "changes",
    [
        {"labelIndex": -1},
        {"labelIndex": 2},
        {"labelIndex": 0.0},
        {"labelIndex": False},
        {"label": "class-1"},
        {"probabilities": [1.0]},
        {"probabilities": ["0.8", "0.2"]},
        {"probabilities": [float("nan"), 0.2]},
        {"probabilities": [float("inf"), 0.2]},
        {"probabilities": [1.1, -0.1]},
        {"probabilities": [0.8, 0.8]},
        {"logProbabilities": [0.0]},
        {"logProbabilities": [float("nan"), -1.0]},
        {"logProbabilities": [-1.0, -1.0]},
        {"logProbabilities": [float(np.log(0.2)), float(np.log(0.8))]},
    ],
)
def test_corrupt_prediction_records_cannot_produce_reported_metrics(changes):
    record = {
        "slideId": "a",
        "patientId": "a",
        "labelIndex": 0,
        "label": "class-0",
        "probabilities": [0.8, 0.2],
        **changes,
    }
    with pytest.raises(ValueError):
        classification_metrics([record], target(unit="slide"))


def test_patient_objective_uses_explicit_loss_weights(tmp_path, monkeypatch):
    plan = tiny_plan(tmp_path)
    module = MILTrainModule(4, plan["target"], plan["recipe"])
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    monkeypatch.setattr(module, "forward", lambda *_args: logits)
    monkeypatch.setattr(module, "log", lambda *_args, **_kwargs: None)
    batch = {
        "features": None,
        "mask": None,
        "labels": torch.tensor([0, 0]),
        "lossWeights": torch.tensor([1.5, 0.5]),
    }
    expected = (
        torch.nn.functional.cross_entropy(logits, batch["labels"], reduction="none")
        * batch["lossWeights"]
    ).mean()
    torch.testing.assert_close(module.training_step(batch, 0), expected)


@pytest.mark.parametrize("classes", [2, 3])
def test_actual_cpu_fit_produces_best_last_and_assessment_artifacts(tmp_path, classes):
    plan = tiny_plan(tmp_path, classes)
    output = tmp_path / "run"
    result = train_fold(plan, output)
    assert result["state"] == "succeeded"
    assert (output / "best.ckpt").is_file() and (output / "last.ckpt").is_file()
    history = json.loads((output / "history.json").read_text())
    assert len(history) == 2
    assert all("assessment" not in epoch for epoch in history)
    progress = json.loads((output / "progress.json").read_text())
    assert progress["epoch"] == progress["maxEpochs"] == 2
    assert progress["globalStep"] == history[-1]["step"]
    assert progress["validation"] == history[-1]["validation"]
    assert "assessment" not in progress
    assert result["bestValidationScore"] == pytest.approx(
        min(epoch["validation"]["loss"] for epoch in history)
    )
    predictions = json.loads((output / "assessment-predictions.json").read_text())
    assert len(predictions["records"]) == classes * 2
    assert {row["slideId"] for row in predictions["records"]} == {
        row["slideId"] for row in plan["data"]["memberships"] if row["partition"] == "test"
    }
    assert all(len(row["probabilities"]) == classes for row in predictions["records"])
    assert result["metrics"]["validation"]["selected"]["loss"] == pytest.approx(
        result["bestValidationScore"], rel=1e-5
    )
    checkpoint = torch.load(output / "last.ckpt", map_location="cpu", weights_only=True)
    assert checkpoint["optimizer_states"]
    assert checkpoint["trainingRngState"]["torch"] is not None
    assert checkpoint["trainingRngState"]["cuda"] == []


def test_assessment_labels_cannot_change_training_or_checkpoint_selection(tmp_path):
    plan = tiny_plan(tmp_path)
    altered = copy.deepcopy(plan)
    for row in altered["data"]["memberships"]:
        if row["partition"] == "test":
            row["label"] = "class-1" if row["label"] == "class-0" else "class-0"
    original = train_fold(plan, tmp_path / "original")
    changed = train_fold(altered, tmp_path / "altered")
    assert original["bestValidationScore"] == changed["bestValidationScore"]
    assert json.loads((tmp_path / "original/history.json").read_text()) == json.loads(
        (tmp_path / "altered/history.json").read_text()
    )
    assert (
        original["metrics"]["assessment"]["selected"]["loss"]
        != changed["metrics"]["assessment"]["selected"]["loss"]
    )


def test_legacy_resume_last_checkpoint_continues_optimizer_epoch_and_history(tmp_path):
    plan = tiny_plan(tmp_path)
    plan["recipe"]["maxEpochs"] = 1
    output = tmp_path / "resume"
    first = train_fold(plan, output)
    # Pre-receipt runs retain their original checkpoint-only resume behavior.
    (output / "fit-complete.json").unlink()
    plan["recipe"]["maxEpochs"] = 2
    resumed = train_fold(plan, output, checkpoint_path=first["lastCheckpointPath"])
    history = json.loads((output / "history.json").read_text())
    assert [epoch["epoch"] for epoch in history] == [0, 1]
    assert history[1]["step"] > history[0]["step"]
    assert resumed["resumedFrom"] == first["lastCheckpointPath"]
    assert not resumed["assessmentOnlyResume"]


@pytest.mark.parametrize("workers", [0, 1, 2])
def test_legacy_epoch_checkpoint_resume_reproduces_uninterrupted_training_with_dropout(
    tmp_path, workers
):
    plan = tiny_plan(tmp_path)
    plan["recipe"].update(dropout=0.25, maxEpochs=3)
    plan["resources"]["dataLoaderWorkers"] = workers
    full = train_fold(copy.deepcopy(plan), tmp_path / "full")
    first_plan = copy.deepcopy(plan)
    first_plan["recipe"]["maxEpochs"] = 1
    first = train_fold(first_plan, tmp_path / "resumed")
    (tmp_path / "resumed/fit-complete.json").unlink()
    resumed = train_fold(plan, tmp_path / "resumed", checkpoint_path=first["lastCheckpointPath"])
    full_state = torch.load(full["lastCheckpointPath"], map_location="cpu", weights_only=True)
    resumed_state = torch.load(resumed["lastCheckpointPath"], map_location="cpu", weights_only=True)
    assert full_state["metricsHistory"] == resumed_state["metricsHistory"]
    for name, value in full_state["state_dict"].items():
        torch.testing.assert_close(value, resumed_state["state_dict"][name], rtol=0, atol=0)


@pytest.mark.parametrize("interrupted_at", ["validation", "assessment", "metrics"])
def test_completed_fit_resumes_assessment_without_another_epoch(
    tmp_path, monkeypatch, interrupted_at
):
    import histopilot.training.fold as fold

    plan = tiny_plan(tmp_path)
    plan["recipe"].update(
        maxEpochs=5, minEpochs=3, earlyStopping=True, patience=1, earlyStoppingMinDelta=1e6
    )
    full = train_fold(copy.deepcopy(plan), tmp_path / "full")
    output = tmp_path / "interrupted"
    original_predict, original_write = fold._predict, fold._write_json
    prediction_calls = 0

    def interrupt_prediction(*args, **kwargs):
        nonlocal prediction_calls
        prediction_calls += 1
        if (interrupted_at, prediction_calls) in {("validation", 1), ("assessment", 2)}:
            raise RuntimeError("Interrupted assessment")
        return original_predict(*args, **kwargs)

    def interrupt_artifact(path, value):
        if interrupted_at == "metrics" and path.name == "metrics.json":
            raise RuntimeError("Interrupted assessment")
        return original_write(path, value)

    monkeypatch.setattr(fold, "_predict", interrupt_prediction)
    monkeypatch.setattr(fold, "_write_json", interrupt_artifact)
    with pytest.raises(RuntimeError, match="Interrupted assessment"):
        train_fold(plan, output)
    evidence = {
        name: (output / name).read_bytes()
        for name in ("best.ckpt", "last.ckpt", "fit-complete.json", "history.json")
    }
    assert len(json.loads(evidence["history.json"])) == 3

    def unexpected_fit(*_args, **_kwargs):
        raise AssertionError("A completed fit must never re-enter optimization.")

    monkeypatch.setattr(fold, "_predict", original_predict)
    monkeypatch.setattr(fold, "_write_json", original_write)
    monkeypatch.setattr(fold.L.Trainer, "fit", unexpected_fit)
    resumed = train_fold(plan, output, checkpoint_path=output / "last.ckpt")
    assert resumed["assessmentOnlyResume"]
    assert resumed["epochsCompleted"] == full["epochsCompleted"] == 3
    assert resumed["bestEpoch"] == full["bestEpoch"]
    assert resumed["bestValidationScore"] == full["bestValidationScore"]
    assert resumed["metrics"] == full["metrics"]
    assert all((output / name).read_bytes() == content for name, content in evidence.items())


@pytest.mark.parametrize(
    "changed", ["best", "last", "missing_checkpoint", "plan", "receipt", "resume_checkpoint"]
)
def test_completed_fit_refuses_changed_evidence_before_loading_or_training(
    tmp_path, monkeypatch, changed
):
    import histopilot.training.fold as fold

    plan = tiny_plan(tmp_path)
    plan["recipe"]["maxEpochs"] = 1
    output = tmp_path / "completed"
    train_fold(plan, output)
    checkpoint = output / "last.ckpt"
    if changed in {"best", "last"}:
        path = output / f"{changed}.ckpt"
        path.write_bytes(path.read_bytes() + b"changed")
    elif changed == "missing_checkpoint":
        checkpoint.unlink()
    elif changed == "plan":
        plan["recipe"]["maxEpochs"] = 2
    elif changed == "receipt":
        path = output / "fit-complete.json"
        receipt = json.loads(path.read_text())
        receipt["fit"]["history"] = []
        path.write_text(json.dumps(receipt))
    else:
        checkpoint = output / "best.ckpt"

    def unexpected_compute(*_args, **_kwargs):
        raise AssertionError("Verify completion evidence before loading or training a model.")

    monkeypatch.setattr(fold.L.Trainer, "fit", unexpected_compute)
    monkeypatch.setattr(MILTrainModule, "load_from_checkpoint", unexpected_compute)
    with pytest.raises(ValueError, match="Cannot resume completed fitting"):
        train_fold(plan, output, checkpoint_path=checkpoint)


def test_patient_overlap_fails_before_training(tmp_path):
    plan = tiny_plan(tmp_path)
    plan["data"]["memberships"][-1]["patientId"] = plan["data"]["memberships"][0]["patientId"]
    with pytest.raises(ValueError, match="overlap"):
        train_fold(plan, tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()


def test_slide_target_can_finish_when_patient_level_labels_are_undefined(tmp_path):
    plan = tiny_plan(tmp_path)
    plan["target"]["unit"] = "slide"
    plan["recipe"]["maxEpochs"] = 1
    for row in plan["data"]["memberships"]:
        row["patientId"] = f"{row['partition']}-patient"
    result = train_fold(plan, tmp_path / "slide-run")
    assert result["metrics"]["assessment"]["selected"]["available"]
    assert not result["metrics"]["assessment"]["patient"]["available"]
    exported = json.loads((tmp_path / "slide-run/assessment-predictions.json").read_text())
    assert exported["patientRecords"] is None


def test_sigterm_at_training_boundary_saves_checkpoint_without_success(tmp_path, monkeypatch):
    plan = tiny_plan(tmp_path)
    original = MILTrainModule.training_step

    def interrupt(module, batch, batch_idx):
        if module.global_step >= 1:
            raise SIGTERMException()
        return original(module, batch, batch_idx)

    monkeypatch.setattr(MILTrainModule, "training_step", interrupt)
    output = tmp_path / "cancelled"
    with pytest.raises(SIGTERMException):
        train_fold(plan, output)
    assert (output / "last.ckpt").is_file()
    assert not (output / "result.json").exists()


@pytest.mark.parametrize("interrupted_step", [1, 3])
def test_cancellation_replays_interrupted_epoch_reproducibly(
    tmp_path, monkeypatch, interrupted_step
):
    plan = tiny_plan(tmp_path)
    plan["recipe"].update(dropout=0.25, maxEpochs=3)
    full = train_fold(copy.deepcopy(plan), tmp_path / "uninterrupted")
    original = MILTrainModule.training_step

    def interrupt(module, batch, batch_idx):
        if module.global_step >= interrupted_step:
            raise SIGTERMException()
        return original(module, batch, batch_idx)

    monkeypatch.setattr(MILTrainModule, "training_step", interrupt)
    output = tmp_path / "interrupted"
    with pytest.raises(SIGTERMException):
        train_fold(copy.deepcopy(plan), output)
    assert not (output / "fit-complete.json").exists()
    saved = torch.load(output / "last.ckpt", map_location="cpu", weights_only=True)
    assert saved["global_step"] == interrupted_step - 1
    monkeypatch.setattr(MILTrainModule, "training_step", original)
    resumed = train_fold(plan, output, checkpoint_path=output / "last.ckpt")
    assert not resumed["assessmentOnlyResume"]
    full_state = torch.load(full["lastCheckpointPath"], map_location="cpu", weights_only=True)
    resumed_state = torch.load(resumed["lastCheckpointPath"], map_location="cpu", weights_only=True)
    assert full_state["metricsHistory"] == resumed_state["metricsHistory"]
    for name, value in full_state["state_dict"].items():
        torch.testing.assert_close(value, resumed_state["state_dict"][name], rtol=0, atol=0)


def test_auprc_uses_declared_positive_class_and_groups_tied_scores():
    rows = [
        {
            "slideId": str(index),
            "patientId": str(index),
            "labelIndex": label,
            "label": f"class-{label}",
            "probabilities": probabilities,
        }
        for index, (label, probabilities) in enumerate(
            [
                (0, [0.8, 0.2]),
                (1, [0.8, 0.2]),
                (0, [0.3, 0.7]),
                (1, [0.1, 0.9]),
            ]
        )
    ]
    metrics = classification_metrics(rows, target())["selected"]
    assert metrics["auprc"] == pytest.approx((0.5 + 2 / 3) / 2)
    assert classification_metrics(rows[::-1], target())["selected"]["auprc"] == metrics["auprc"]
    no_positive = [row for row in rows if row["labelIndex"] == 1]
    assert classification_metrics(no_positive, target())["selected"]["auprc"] is None


def test_cosine_warmup_accumulation_and_clipping_resume_without_changing_results(
    tmp_path, monkeypatch
):
    plan = tiny_plan(tmp_path)
    plan["recipe"].update(
        dropout=0.25,
        maxEpochs=4,
        lrScheduler="cosine",
        warmupEpochs=1,
        finalLrFraction=0.01,
        accumulateGradBatches=2,
        gradientClipNorm=0.1,
    )
    full = train_fold(copy.deepcopy(plan), tmp_path / "full-schedule")
    original = MILTrainModule.training_step

    def interrupt(module, batch, batch_idx):
        if module.current_epoch == 2 and batch_idx == 0:
            raise SIGTERMException()
        return original(module, batch, batch_idx)

    monkeypatch.setattr(MILTrainModule, "training_step", interrupt)
    output = tmp_path / "interrupted-schedule"
    with pytest.raises(SIGTERMException):
        train_fold(copy.deepcopy(plan), output)
    monkeypatch.setattr(MILTrainModule, "training_step", original)
    resumed = train_fold(plan, output, checkpoint_path=output / "last.ckpt")
    full_state = torch.load(full["lastCheckpointPath"], map_location="cpu", weights_only=True)
    resumed_state = torch.load(resumed["lastCheckpointPath"], map_location="cpu", weights_only=True)
    assert full_state["metricsHistory"] == resumed_state["metricsHistory"]
    assert full_state["lr_schedulers"] == resumed_state["lr_schedulers"]
    assert full_state["global_step"] == 4
    assert [epoch["learningRate"] for epoch in full_state["metricsHistory"]] == pytest.approx(
        [
            0.01,
            0.02,
            0.0101,
            0.0002,
        ]
    )
    for name, value in full_state["state_dict"].items():
        torch.testing.assert_close(value, resumed_state["state_dict"][name], rtol=0, atol=0)


def test_early_stopping_minimum_epochs_preserves_best_checkpoint_selection(tmp_path):
    plan = tiny_plan(tmp_path)
    plan["recipe"].update(
        maxEpochs=5,
        minEpochs=3,
        earlyStopping=True,
        patience=1,
        earlyStoppingMinDelta=1e6,
    )
    result = train_fold(plan, tmp_path / "minimum-epochs")
    assert result["epochsCompleted"] == 3
    history = json.loads((tmp_path / "minimum-epochs/history.json").read_text())
    assert result["bestValidationScore"] == pytest.approx(
        min(epoch["validation"]["loss"] for epoch in history)
    )


def test_early_patience_exhaustion_can_recover_before_epoch_floor_and_resume(tmp_path, monkeypatch):
    from lightning.pytorch.callbacks import EarlyStopping

    plan = tiny_plan(tmp_path)
    plan["recipe"].update(maxEpochs=7, minEpochs=5, earlyStopping=True, patience=1, dropout=0.25)
    scores = [0.6, 0.7, 0.8, 0.5, 0.4, 0.4, 0.4]
    original_validation_end = MILTrainModule.on_validation_epoch_end
    original_step = MILTrainModule.training_step

    def prescribed_validation(module):
        probability = float(np.exp(-scores[module.current_epoch]))
        for row in module.validation_rows:
            row["probabilities"] = [
                probability if index == row["labelIndex"] else 1 - probability for index in range(2)
            ]
            row.pop("logProbabilities", None)
        original_validation_end(module)

    monkeypatch.setattr(MILTrainModule, "on_validation_epoch_end", prescribed_validation)
    full = train_fold(copy.deepcopy(plan), tmp_path / "recover-full")
    assert full["epochsCompleted"] == 6
    assert full["bestValidationScore"] == pytest.approx(0.4)

    def interrupt(module, batch, batch_idx):
        if module.current_epoch == 3:
            raise SIGTERMException()
        return original_step(module, batch, batch_idx)

    monkeypatch.setattr(MILTrainModule, "training_step", interrupt)
    output = tmp_path / "recover-resume"
    with pytest.raises(SIGTERMException):
        train_fold(copy.deepcopy(plan), output)
    saved = torch.load(output / "last.ckpt", map_location="cpu", weights_only=True)
    assert saved["epoch"] == 2
    assert [epoch["epoch"] for epoch in saved["metricsHistory"]] == [0, 1, 2]
    callback_key = EarlyStopping("validation_loss", mode="min").state_key
    assert saved["callbacks"][callback_key]["wait_count"] == 2
    assert saved["callbacks"][callback_key]["best_score"] == pytest.approx(0.6)
    assert saved["callbacks"][callback_key]["stopping_reason"] == 0
    monkeypatch.setattr(MILTrainModule, "training_step", original_step)
    resumed = train_fold(plan, output, checkpoint_path=output / "last.ckpt")
    assert resumed["epochsCompleted"] == 6
    full_state = torch.load(full["lastCheckpointPath"], map_location="cpu", weights_only=True)
    resumed_state = torch.load(resumed["lastCheckpointPath"], map_location="cpu", weights_only=True)
    assert full_state["metricsHistory"] == resumed_state["metricsHistory"]
    assert full_state["callbacks"][callback_key] == resumed_state["callbacks"][callback_key]
    best = torch.load(resumed["bestCheckpointPath"], map_location="cpu", weights_only=True)
    assert best["epoch"] == 4
    for name, value in full_state["state_dict"].items():
        torch.testing.assert_close(value, resumed_state["state_dict"][name], rtol=0, atol=0)


def test_bf16_cpu_fit_and_selected_checkpoint_prediction_use_same_precision(tmp_path):
    plan = tiny_plan(tmp_path)
    plan["recipe"].update(maxEpochs=1, precision="bf16-mixed")
    result = train_fold(plan, tmp_path / "bf16")
    assert result["precision"] == "bf16-mixed"
    assert result["metrics"]["validation"]["selected"]["loss"] == pytest.approx(
        result["bestValidationScore"],
        rel=1e-6,
    )


def test_fp16_cpu_is_rejected_without_training(tmp_path):
    plan = tiny_plan(tmp_path)
    plan["recipe"]["precision"] = "16-mixed"
    with pytest.raises(ValueError, match="requires a CUDA GPU"):
        train_fold(plan, tmp_path / "fp16-cpu")
    assert not (tmp_path / "fp16-cpu/last.ckpt").exists()


def test_undefined_auroc_monitor_fails_before_fitting(tmp_path):
    plan = tiny_plan(tmp_path)
    plan["recipe"]["checkpointMetric"] = "validation_auroc"
    for row in plan["data"]["memberships"]:
        if row["partition"] == "val":
            row["label"] = "class-0"
    with pytest.raises(ValueError, match="Validation AUROC cannot"):
        train_fold(plan, tmp_path / "undefined-monitor")
    assert not (tmp_path / "undefined-monitor").exists()


def test_nonfinite_gradients_fail_before_optimizer_update_and_preserve_initial_checkpoint(
    tmp_path, monkeypatch
):
    plan = tiny_plan(tmp_path)
    original = MILTrainModule.on_train_start

    def inject_bad_gradient(module):
        original(module)
        next(module.parameters()).register_hook(lambda gradient: gradient * float("nan"))

    monkeypatch.setattr(MILTrainModule, "on_train_start", inject_bad_gradient)
    output = tmp_path / "bad-gradient"
    with pytest.raises(FloatingPointError, match="Nonfinite training gradients"):
        train_fold(plan, output)
    checkpoint = torch.load(output / "last.ckpt", map_location="cpu", weights_only=True)
    assert checkpoint["global_step"] == 0
    assert all(bool(torch.isfinite(value).all()) for value in checkpoint["state_dict"].values())
    assert not (output / "result.json").exists()


def test_cpu_checkpoint_rng_does_not_initialize_or_restore_cuda_contexts(tmp_path, monkeypatch):
    plan = tiny_plan(tmp_path)
    module = MILTrainModule(4, plan["target"], plan["recipe"])

    def unexpected_cuda_rng(*_args):
        raise AssertionError("A CPU training checkpoint must not touch CUDA RNG state.")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", unexpected_cuda_rng)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", unexpected_cuda_rng)
    checkpoint = {}
    module.on_save_checkpoint(checkpoint)
    assert checkpoint["trainingRngState"]["cuda"] == []
    checkpoint["trainingRngState"]["cuda"] = [torch.tensor([1], dtype=torch.uint8)]
    module.on_load_checkpoint(checkpoint)
    module.on_train_start()
    assert module._resume_rng_state is None


def test_interrupted_fit_shuts_down_both_cached_worker_pools(tmp_path, monkeypatch):
    from histopilot.datasets.datamodule import MILDataModule

    plan = tiny_plan(tmp_path)
    plan["resources"]["dataLoaderWorkers"] = 1
    original_step = MILTrainModule.training_step
    original_teardown = MILDataModule.teardown
    closed_workers = {}

    def interrupt(module, batch, batch_idx):
        if module.current_epoch == 1:
            # Lightning may reconstruct loaders while attaching its sampler;
            # capture the actual running pools before its exception teardown.
            for role, loader in (
                ("train", module.trainer.train_dataloader),
                ("val", module.trainer.val_dataloaders),
            ):
                if isinstance(loader, list):
                    loader = loader[0]
                closed_workers.setdefault(role, []).extend(loader._iterator._workers)
            raise SIGTERMException()
        return original_step(module, batch, batch_idx)

    def close_and_check(module, stage=None):
        for role, loader in module._loaders.items():
            if loader._iterator is not None:
                closed_workers.setdefault(role, []).extend(loader._iterator._workers)
        original_teardown(module, stage)
        assert not module._loaders
        assert all(
            not worker.is_alive() for workers in closed_workers.values() for worker in workers
        )

    monkeypatch.setattr(MILTrainModule, "training_step", interrupt)
    monkeypatch.setattr(MILDataModule, "teardown", close_and_check)
    with pytest.raises(SIGTERMException):
        train_fold(plan, tmp_path / "worker-cleanup")
    assert set(closed_workers) == {"train", "val"}
