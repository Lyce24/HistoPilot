"""Train a fresh ABMIL model on all development slides for a reviewed epoch budget."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from histopilot.datasets.datamodule import MILDataModule
from histopilot.datasets.mil import EpochShuffleSampler, MILDataError, SlideDataset
from histopilot.schemas.development import TrainingRecipe
from histopilot.training.fold import _HistoryWriter, _write_json
from histopilot.training.module import MILTrainModule


class RefitDataModule(MILDataModule):
    """A train-only loader; never creates a validation or test dataset."""

    def __init__(self, plan):
        L.LightningDataModule.__init__(self)
        self.plan = deepcopy({**plan, **plan["data"]})
        rows, target = self.plan["memberships"], self.plan["target"]
        if not rows or len({row["slideId"] for row in rows}) != len(rows):
            raise MILDataError("Refit requires each development slide exactly once.")
        classes, patient_labels = target["classes"], {}
        if len(classes) < 2 or len(set(classes)) != len(classes):
            raise MILDataError("Refit requires an explicit ordered class contract.")
        for row in rows:
            if (
                row.get("partition") != "train"
                or row.get("pool") != "development"
                or row.get("phase") != "refit"
                or not row.get("patientId")
                or row.get("label") not in classes
            ):
                raise MILDataError("Refit accepts only frozen development training memberships.")
            if (
                target["unit"] == "patient"
                and row["patientId"] in patient_labels
                and patient_labels[row["patientId"]] != row["label"]
            ):
                raise MILDataError("Patient labels conflict in the development refit pool.")
            patient_labels[row["patientId"]] = row["label"]
            row["labelIndex"] = classes.index(row["label"])
        if {row["label"] for row in rows} != set(classes):
            raise MILDataError("Every frozen target class must appear in refit training.")
        self.memberships = {"train": sorted(rows, key=lambda row: row["slideId"])}
        self.batch_size = self.plan["recipe"]["batchSize"]
        self.num_workers = self.plan["resources"].get("dataLoaderWorkers", 0)
        if type(self.num_workers) is not int or not 0 <= self.num_workers <= 64:
            raise MILDataError("Invalid refit data loader worker count.")
        self.train_dataset = self.val_dataset = self.test_dataset = None
        self.train_sampler = None
        self._loaders = {}
        self.epoch = 0

    def setup(self, stage=None):
        if stage not in (None, "fit"):
            raise MILDataError("A refit does not use validation or test data.")
        if self.train_dataset is None:
            self.train_dataset = SlideDataset(self.plan, self.memberships["train"], training=True)
            self.train_sampler = EpochShuffleSampler(self.train_dataset)
        self.set_epoch(self.epoch)

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        raise MILDataError("Test inference runs separately after model publication.")


class _RefitHistory(_HistoryWriter):
    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.history.append(
            {
                "epoch": int(trainer.current_epoch),
                "step": int(trainer.global_step),
                "trainingLoss": pl_module._training_loss_sum / max(1, pl_module._training_count),
                "learningRate": pl_module._epoch_learning_rate,
            }
        )
        _write_json(self.path, pl_module.history)
        _write_json(
            self.path.parent / "progress.json",
            {
                **pl_module.history[-1],
                "epoch": int(trainer.current_epoch) + 1,
                "maxEpochs": trainer.max_epochs,
                "validation": None,
            },
        )


def train_refit(plan, output_dir, *, checkpoint_path=None):
    """Fixed-epoch fitting from fresh weights; resume replays incomplete epochs."""
    recipe = TrainingRecipe.model_validate(plan["recipe"], context={"legacy": True}).model_dump()
    epochs = plan["epochBudget"]["epochs"]
    if recipe["maxEpochs"] != epochs or recipe["minEpochs"] != epochs or recipe["earlyStopping"]:
        raise ValueError("Refit must use its reviewed fixed epoch budget with no early stopping.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = plan.get("device", "cpu")
    if device not in {"cpu", "cuda"}:
        raise ValueError("A refit device must be CPU or its assigned GPU.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The selected GPU is unavailable.")
    if recipe["precision"] == "16-mixed" and device != "cuda":
        raise ValueError("FP16 training requires a GPU.")
    if (
        recipe["precision"] == "bf16-mixed"
        and device == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("The selected GPU does not support BF16.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(plan["resources"].get("cpuThreadsPerRun", 2))
    L.seed_everything(plan["trainingSeed"], workers=True)
    datamodule = RefitDataModule({**plan, "recipe": recipe})
    model = MILTrainModule(plan["data"]["featureDim"], plan["target"], recipe)
    checkpoint = ModelCheckpoint(
        dirpath=output_dir,
        save_top_k=0,
        save_last=True,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        enable_version_counter=False,
    )
    trainer = L.Trainer(
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        max_epochs=epochs,
        min_epochs=epochs,
        callbacks=[_RefitHistory(output_dir / "history.json"), checkpoint],
        logger=CSVLogger(str(output_dir), name="logs"),
        deterministic=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        log_every_n_steps=1,
        precision=recipe["precision"],
        gradient_clip_val=recipe["gradientClipNorm"],
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=recipe["accumulateGradBatches"],
    )
    try:
        trainer.fit(
            model,
            datamodule=datamodule,
            ckpt_path=str(checkpoint_path) if checkpoint_path else None,
        )
        if trainer.interrupted or len(model.history) != epochs:
            raise RuntimeError("Refit stopped before its reviewed epoch budget completed.")
        final = output_dir / "final.ckpt"
        trainer.save_checkpoint(str(final))
        result = {
            "runId": plan["runId"],
            "state": "succeeded",
            "bestCheckpointPath": str(final),
            "lastCheckpointPath": str(output_dir / "last.ckpt"),
            "epochsCompleted": epochs,
            "bestEpoch": epochs,
            "epochBudget": plan["epochBudget"],
            "historyPath": str(output_dir / "history.json"),
            "trainingSlideCount": len(datamodule.memberships["train"]),
            "trainingPatientCount": len(
                {row["patientId"] for row in datamodule.memberships["train"]}
            ),
            "trainingObjective": "patient_balanced_slide_cross_entropy"
            if plan["target"]["unit"] == "patient"
            else "slide_cross_entropy",
            "resumedFrom": str(checkpoint_path) if checkpoint_path else None,
            "resumePolicy": "replay_interrupted_epoch_from_last_completed_epoch",
            "validationUsed": False,
            "testDataUsed": False,
        }
        _write_json(output_dir / "result.json", result)
        return result
    finally:
        datamodule.teardown()
