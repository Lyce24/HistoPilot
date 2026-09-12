"""Execute one reviewed training run in an isolated worker process."""

import hashlib
import json
import os
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStoppingReason
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities.exceptions import SIGTERMException

from histopilot.datasets.datamodule import MILDataModule
from histopilot.schemas.development import TrainingRecipe
from histopilot.storage.project_lock import _reject_symlink_components
from histopilot.storage.scientific import ScientificStore
from histopilot.training.module import (
    MILTrainModule,
    aggregate_patients,
    classification_metrics,
    prediction_rows,
)
from histopilot.workers.packing_process import write_json as _write_durable_json


def _write_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _receipt_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _checkpoint_evidence(path, output_dir):
    path = Path(path).absolute()
    _reject_symlink_components(path)
    if path.parent != output_dir.absolute() or path.suffix != ".ckpt":
        raise ValueError("A completed fit checkpoint must remain in its original run directory.")
    before = path.stat()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    if any(
        getattr(before, key) != getattr(after, key)
        for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    ):
        raise ValueError("A completed fit checkpoint changed during verification.")
    return {"path": str(path), "bytes": after.st_size, "sha256": digest}


def _completed_fit(plan, output_dir, recipe, checkpoint_path):
    path = output_dir / "fit-complete.json"
    if not path.exists() and not path.is_symlink():
        # Legacy runs have no completion receipt and retain checkpoint-based
        # fitting resume; they cannot distinguish fitting from interrupted assessment.
        return None
    try:
        receipt = json.loads(ScientificStore._read_file(path, 64 * 1024 * 1024))
        payload = receipt["fit"]
        identity = {"plan": plan, "recipe": recipe, "outputPath": str(output_dir.absolute())}
        if (
            receipt["version"] != 1
            or receipt["sha256"] != _receipt_hash(payload)
            or payload["planHash"] != _receipt_hash(identity)
        ):
            raise ValueError("The completed fit receipt or its execution plan changed.")
        if (
            checkpoint_path is None
            or str(Path(checkpoint_path).absolute()) != payload["last"]["path"]
        ):
            raise ValueError("Resume a completed fit with its recorded last checkpoint.")
        for key in ("best", "last"):
            if _checkpoint_evidence(payload[key]["path"], output_dir) != payload[key]:
                raise ValueError("A completed fit checkpoint changed after fitting.")
        return payload
    except (KeyError, TypeError, OSError, ValueError) as error:
        raise ValueError(f"Cannot resume completed fitting: {error}") from error


class _MinimumEpochEarlyStopping(EarlyStopping):
    """Track patience from epoch zero, but publish patience stops after the floor.

    Lightning's trainer keeps ``should_stop`` sticky even when ``min_epochs``
    postpones stopping. Gating the decision here lets a later improvement reset
    patience during that window, without clearing another callback's stop.
    """

    _minimum_epochs_reached = True

    @property
    def state_key(self):
        # Preserve the original callback key and inherited best/wait state so
        # checkpoint restoration retains the complete early-stopping history.
        return f"EarlyStopping{dict(monitor=self.monitor, mode=self.mode)!r}"

    def _run_early_stopping_check(self, trainer):
        self._minimum_epochs_reached = trainer.current_epoch + 1 >= (trainer.min_epochs or 0)
        super()._run_early_stopping_check(trainer)

    def _evaluate_stopping_criteria(self, current):
        should_stop, reason = super()._evaluate_stopping_criteria(current)
        if (
            should_stop
            and not self._minimum_epochs_reached
            and self.stopping_reason == EarlyStoppingReason.PATIENCE_EXHAUSTED
        ):
            self.stopping_reason = EarlyStoppingReason.NOT_STOPPED
            return False, None
        return should_stop, reason


class _EveryEpochCheckpoint(ModelCheckpoint):
    """Keep the latest completed epoch even when the monitored score worsens."""

    @property
    def state_key(self):
        return super().state_key.replace(type(self).__qualname__, "ModelCheckpoint", 1)

    def on_validation_end(self, trainer, pl_module):
        eligible = not self._should_skip_saving_checkpoint(
            trainer
        ) and not self._should_save_on_train_epoch_end(trainer)
        super().on_validation_end(trainer, pl_module)
        # Lightning 2.6 updates save_last=True only when top-k actually saved.
        # A resumable checkpoint needs the optimizer, RNG, history and patience
        # from this complete epoch even if best.ckpt remains an earlier epoch.
        if eligible and self._last_global_step_saved != trainer.global_step:
            self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))


class _HistoryWriter(L.Callback):
    def __init__(self, path):
        self.path = path

    def on_train_start(self, trainer, pl_module):
        initial = self.path.parent / "last.ckpt"
        if not initial.exists():
            # A cancellation during epoch zero can restart reproducibly from
            # initialized parameters and optimizer state before any update.
            trainer.save_checkpoint(str(initial))

    def on_validation_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            _write_json(self.path, pl_module.history)
            if pl_module.history:
                latest = pl_module.history[-1]
                _write_json(
                    self.path.parent / "progress.json",
                    {
                        "epoch": int(trainer.current_epoch) + 1,
                        "maxEpochs": trainer.max_epochs,
                        "globalStep": int(trainer.global_step),
                        "trainingLoss": latest["trainingLoss"],
                        "learningRate": latest["learningRate"],
                        "validation": latest["validation"],
                        "cudaPeakAllocatedBytes": torch.cuda.max_memory_allocated()
                        if pl_module.device.type == "cuda"
                        else 0,
                        "cudaPeakReservedBytes": torch.cuda.max_memory_reserved()
                        if pl_module.device.type == "cuda"
                        else 0,
                    },
                )

    def on_exception(self, trainer, pl_module, exception):
        if isinstance(exception, (KeyboardInterrupt, SIGTERMException)):
            # Keep the last complete epoch (or initialization) checkpoint.
            # DataLoader prefetch/cursors are not resumable; saving halfway
            # through an epoch would silently repeat or skip selected slides.
            _write_json(self.path, pl_module.history)


def _validate_plan(plan):
    rows = plan["data"]["memberships"]
    partitions = {"train": set(), "val": set(), "test": set()}
    slides = set()
    for row in rows:
        if row["partition"] not in partitions:
            raise ValueError(
                "The ABMIL worker accepts training, validation and development assessment only."
            )
        if row["slideId"] in slides:
            raise ValueError("A run cannot contain the same slide more than once.")
        slides.add(row["slideId"])
        if not row.get("patientId"):
            raise ValueError("Every training-plan slide requires a frozen grouping identity.")
        partitions[row["partition"]].add(row["patientId"])
        if row["label"] not in plan["target"]["classes"]:
            raise ValueError("A run label is outside the frozen target classes.")
        if row.get("phase") == "final" or row.get("pool") == "external_test":
            raise ValueError("Reserved final cohorts cannot enter a development training run.")
    if not partitions["train"] or not partitions["val"]:
        raise ValueError("A run needs nonempty training and validation partitions.")
    if any(
        partitions[left] & partitions[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise ValueError(
            "Patient groups overlap between the run's fitting, validation or assessment partitions."
        )
    if plan["recipe"].get("checkpointMetric") == "validation_auroc":
        validation_classes = {row["label"] for row in rows if row["partition"] == "val"}
        if validation_classes != set(plan["target"]["classes"]):
            raise ValueError(
                "Validation AUROC cannot select checkpoints when a frozen class is absent. "
                "Choose validation loss or revise the development split before freezing."
            )


def _predict(model, loader, target, device, precision="32-true"):
    rows = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if precision == "16-mixed" else torch.bfloat16,
                enabled=precision != "32-true",
            ):
                logits = model(batch["features"].to(device), batch["mask"].to(device))
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError(
                    "The selected checkpoint produced nonfinite probabilities."
                )
            rows.extend(prediction_rows(batch, logits, target))
    return sorted(rows, key=lambda row: row["slideId"])


def train_fold(plan: dict, output_dir: Path, *, checkpoint_path=None) -> dict:
    """Fit one plan, resume a last checkpoint, and assess its best validation checkpoint.

    The scheduler assigns CUDA visibility before this module is imported. Data
    memberships and class order are inherited verbatim from the frozen protocol.
    No external cohort or assessment metric participates in checkpoint selection.
    A verified completion receipt resumes assessment without further fitting.
    Legacy runs without a receipt retain their original checkpoint resume behavior.
    """
    _validate_plan(plan)
    recipe = TrainingRecipe.model_validate(plan["recipe"]).model_dump()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resources = plan["resources"]
    device_name = plan.get("device", "cpu")
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("A worker device must be CPU or its scheduler-assigned CUDA device.")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The assigned CUDA device is unavailable in this training runtime.")
    if recipe["precision"] == "16-mixed" and device_name != "cuda":
        raise ValueError("16-bit mixed precision requires a CUDA GPU; use 32-bit or BF16 on CPU.")
    if (
        recipe["precision"] == "bf16-mixed"
        and device_name == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("The assigned CUDA GPU does not support BF16 mixed precision.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(resources.get("cpuThreadsPerRun", 2))
    L.seed_everything(plan["trainingSeed"], workers=True)
    if device_name == "cuda":
        torch.cuda.reset_peak_memory_stats()
    datamodule = MILDataModule({**plan, **plan["data"], "recipe": recipe})
    try:
        return _fit_and_assess(plan, output_dir, recipe, datamodule, checkpoint_path)
    finally:
        # Fit exceptions and interrupted prediction must not leave worker pools
        # and memory maps alive in the calling process.
        datamodule.teardown()


def _fit(plan, output_dir, recipe, datamodule, checkpoint_path):
    target = plan["target"]
    device_name = plan.get("device", "cpu")
    model = MILTrainModule(plan["data"]["featureDim"], target, recipe)
    monitor = recipe["checkpointMetric"]
    mode = "min" if monitor == "validation_loss" else "max"
    checkpoint = _EveryEpochCheckpoint(
        dirpath=output_dir,
        filename="best",
        monitor=monitor,
        mode=mode,
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
        enable_version_counter=False,
        save_on_train_epoch_end=False,
    )
    callbacks = [checkpoint, _HistoryWriter(output_dir / "history.json")]
    if recipe.get("earlyStopping", True):
        callbacks.append(
            _MinimumEpochEarlyStopping(
                monitor=monitor,
                mode=mode,
                patience=recipe["patience"],
                min_delta=recipe["earlyStoppingMinDelta"],
                check_finite=True,
                check_on_train_epoch_end=False,
            )
        )
    trainer = L.Trainer(
        accelerator="gpu" if device_name == "cuda" else "cpu",
        devices=1,
        max_epochs=recipe["maxEpochs"],
        min_epochs=recipe["minEpochs"],
        callbacks=callbacks,
        logger=CSVLogger(str(output_dir), name="logs"),
        deterministic=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        precision=recipe["precision"],
        gradient_clip_val=recipe["gradientClipNorm"],
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=recipe["accumulateGradBatches"],
    )
    trainer.fit(
        model, datamodule=datamodule, ckpt_path=str(checkpoint_path) if checkpoint_path else None
    )
    if trainer.interrupted:
        raise RuntimeError("Training was interrupted; resume its last checkpoint to continue.")
    best = Path(checkpoint.best_model_path)
    last = Path(checkpoint.last_model_path)
    if (
        not checkpoint.best_model_path
        or not best.is_file()
        or not checkpoint.last_model_path
        or not last.is_file()
    ):
        raise RuntimeError("Training ended without both best and resumable last checkpoints.")
    fit = {
        "planHash": _receipt_hash(
            {"plan": plan, "recipe": recipe, "outputPath": str(output_dir.absolute())}
        ),
        "best": _checkpoint_evidence(best, output_dir),
        "last": _checkpoint_evidence(last, output_dir),
        "history": model.history,
        "bestValidationScore": float(checkpoint.best_model_score.cpu()),
        "cudaPeakAllocatedBytes": torch.cuda.max_memory_allocated() if device_name == "cuda" else 0,
        "cudaPeakReservedBytes": torch.cuda.max_memory_reserved() if device_name == "cuda" else 0,
    }
    _write_durable_json(
        output_dir / "fit-complete.json",
        {"version": 1, "fit": fit, "sha256": _receipt_hash(fit)},
    )
    return fit


def _fit_and_assess(plan, output_dir, recipe, datamodule, checkpoint_path):
    target = plan["target"]
    device_name = plan.get("device", "cpu")
    fit = _completed_fit(plan, output_dir, recipe, checkpoint_path)
    assessment_only = fit is not None
    if fit is None:
        fit = _fit(plan, output_dir, recipe, datamodule, checkpoint_path)
    best, last = Path(fit["best"]["path"]), Path(fit["last"]["path"])
    _write_json(output_dir / "history.json", fit["history"])
    selected = MILTrainModule.load_from_checkpoint(str(best), map_location="cpu", weights_only=True)
    device = torch.device(device_name)
    selected.to(device)
    validation = _predict(
        selected, datamodule.val_dataloader(), target, device, recipe["precision"]
    )
    # Assessment data is loaded for the first time only after model fitting.
    datamodule.setup("test")
    assessment = _predict(
        selected, datamodule.test_dataloader(), target, device, recipe["precision"]
    )
    predictions = {}
    metrics = {}
    for split, records in (("validation", validation), ("assessment", assessment)):
        metrics[split] = classification_metrics(records, target)
        patient_available = metrics[split]["patient"]["available"]
        path = output_dir / f"{split}-predictions.json"
        _write_json(
            path,
            {
                "classOrder": target["classes"],
                "unit": target["unit"],
                "checkpointPath": str(best),
                "records": records,
                "patientRecords": aggregate_patients(records) if patient_available else None,
                "patientMetrics": metrics[split]["patient"],
            },
        )
        predictions[split] = str(path)
    _write_json(output_dir / "metrics.json", metrics)
    result = {
        "runId": plan["runId"],
        "state": "succeeded",
        "bestCheckpointPath": str(best),
        "lastCheckpointPath": str(last),
        "metricsPath": str(output_dir / "metrics.json"),
        "historyPath": str(output_dir / "history.json"),
        "predictions": predictions,
        "metrics": metrics,
        "checkpointMetric": recipe["checkpointMetric"],
        "checkpointUnit": target["unit"],
        "bestValidationScore": fit["bestValidationScore"],
        "bestEpoch": int(selected.history[-1]["epoch"]) + 1,
        "epochsCompleted": len(fit["history"]),
        "trainingObjective": "patient_balanced_slide_cross_entropy"
        if target["unit"] == "patient"
        else "slide_cross_entropy",
        "resumedFrom": str(checkpoint_path) if checkpoint_path else None,
        "assessmentOnlyResume": assessment_only,
        "resumePolicy": "replay_interrupted_epoch_from_last_completed_epoch",
        "precision": recipe["precision"],
        "accumulateGradBatches": recipe["accumulateGradBatches"],
        "lrScheduler": recipe["lrScheduler"],
        "cudaPeakAllocatedBytes": max(
            fit["cudaPeakAllocatedBytes"],
            torch.cuda.max_memory_allocated() if device_name == "cuda" else 0,
        ),
        "cudaPeakReservedBytes": max(
            fit["cudaPeakReservedBytes"],
            torch.cuda.max_memory_reserved() if device_name == "cuda" else 0,
        ),
    }
    _write_json(output_dir / "result.json", result)
    return result
