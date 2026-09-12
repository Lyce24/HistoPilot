"""Whole-bag inference for frozen single-refit and mean-probability ensembles."""

import csv
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from histopilot.application.feature_bundles import _hash
from histopilot.application.predictors import checkpoint_snapshot
from histopilot.datasets.datamodule import _worker_init
from histopilot.datasets.mil import SlideDataset, collate_mil
from histopilot.storage.project_lock import _reject_symlink_components, ensure_managed_directory
from histopilot.storage.scientific import ScientificStore
from histopilot.training.module import MILTrainModule, _metrics
from histopilot.workers.packing_process import write_json
from histopilot.workers.train_batch import _check_inputs


def _decisions(records, target, threshold):
    for row in records:
        if target["task"] == "binary_classification":
            positive = target["classes"].index(target["positiveClass"])
            predicted = positive if row["probabilities"][positive] >= threshold else 1 - positive
        else:
            predicted = int(np.argmax(row["probabilities"]))
        row.update(predictedIndex=predicted, predictedLabel=target["classes"][predicted])
    return records


def evaluation_metrics(records, target, threshold):
    """Metrics use labeled rows only; threshold changes decisions, never ranking scores."""
    labeled = [row for row in records if row["labelIndex"] is not None]
    result = _metrics(labeled, target)
    if result["available"]:
        confusion = np.zeros((len(target["classes"]), len(target["classes"])), dtype=np.int64)
        for row in labeled:
            confusion[row["labelIndex"], row["predictedIndex"]] += 1
        support = confusion.sum(axis=1)
        recall = np.divide(
            confusion.diagonal(), support, out=np.zeros(len(support)), where=support > 0
        )
        precision = np.divide(
            confusion.diagonal(),
            confusion.sum(axis=0),
            out=np.zeros(len(support)),
            where=confusion.sum(axis=0) > 0,
        )
        f1 = np.divide(
            2 * recall * precision,
            recall + precision,
            out=np.zeros(len(support)),
            where=(recall + precision) > 0,
        )
        result.update(
            accuracy=float(confusion.trace() / len(labeled)),
            balancedAccuracy=float(recall[support > 0].mean()),
            macroF1=float(f1.mean()),
            confusionMatrix=confusion.tolist(),
        )
    return {
        **result,
        "predictionCount": len(records),
        "unlabeledCount": len(records) - len(labeled),
    }


def patient_predictions(records):
    groups = defaultdict(list)
    for row in records:
        if row["patientId"]:
            groups[row["patientId"]].append(row)
    patients = []
    for patient, slides in sorted(groups.items()):
        labels = {row["labelIndex"] for row in slides if row["labelIndex"] is not None}
        if len(labels) > 1:
            raise ValueError("Patient evaluation requires consistent labels within each patient.")
        labeled = next((row for row in slides if row["labelIndex"] is not None), None)
        patients.append(
            {
                "patientId": patient,
                "slideIds": [row["slideId"] for row in slides],
                "labelIndex": labeled["labelIndex"] if labeled else None,
                "label": labeled["label"] if labeled else None,
                "probabilities": np.mean([row["probabilities"] for row in slides], axis=0).tolist(),
            }
        )
        if all("logProbabilities" in row for row in slides):
            patients[-1]["logProbabilities"] = (
                np.logaddexp.reduce([row["logProbabilities"] for row in slides], axis=0)
                - np.log(len(slides))
            ).tolist()
    return patients


def _write_csv(path, rows, classes, *, patient=False):
    fields = (
        (["patientId", "slideIds"] if patient else ["slideId", "patientId"])
        + ["label", "predictedLabel"]
        + [f"probability:{label}" for label in classes]
    )
    _reject_symlink_components(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            value = {key: row.get(key) for key in fields[:4]}
            if patient:
                value["slideIds"] = json.dumps(row["slideIds"], ensure_ascii=False)
            value.update(
                {
                    f"probability:{label}": probability
                    for label, probability in zip(classes, row["probabilities"], strict=True)
                }
            )
            writer.writerow(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def evaluate(plan, output_dir):
    """Run each model sequentially so ensembles do not multiply GPU model memory.

    Completed member predictions are cached with a digest and the full input
    fingerprint. Resume restarts the interrupted member and reuses completed ones.
    """
    folder = Path(output_dir)
    ensure_managed_directory(folder)
    target, data = plan["target"], plan["data"]
    classes = target["classes"]
    method = plan.get("method", "ensemble")
    checkpoints = plan["checkpoints"]
    if (
        method not in {"ensemble", "refit"}
        or not checkpoints
        or (method == "refit" and len(checkpoints) != 1)
    ):
        raise ValueError(
            "A refit requires one checkpoint; an ensemble requires its selected fold checkpoints."
        )
    rows = data["memberships"]
    identities = [row["slideId"] for row in rows]
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("Inference requires exactly one membership per selected slide.")
    if any(row.get("label") is not None and row["label"] not in classes for row in rows):
        raise ValueError("Test labels must preserve the frozen target class order.")
    if target["unit"] == "patient" and any(not row.get("patientId") for row in rows):
        raise ValueError("Patient evaluation requires a grouping identity for every slide.")
    if plan["inference"]["patientAggregation"] != "mean":
        raise ValueError("Frozen predictors require mean patient probabilities.")
    device = torch.device(plan.get("device", "cpu"))
    precision = plan["inference"]["precision"]
    if precision == "float16" and device.type != "cuda":
        raise ValueError("Float16 inference requires CUDA.")
    if device.type == "cuda" and precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("The selected GPU does not support bfloat16 inference.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(plan["resources"]["cpuThreadsPerRun"])
    torch.use_deterministic_algorithms(True)
    _check_inputs(data)
    memberships = [
        {**row, "labelIndex": classes.index(row["label"]) if row.get("label") is not None else -1}
        for row in rows
    ]
    dataset = SlideDataset(
        {**data, "target": target, "trainingSeed": 0, "recipe": {"bagSize": None}},
        memberships,
        training=False,
    )
    workers = plan["inference"]["numWorkers"]
    loader = DataLoader(
        dataset,
        batch_size=plan["inference"]["batchSize"],
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_mil,
        worker_init_fn=_worker_init,
        **({"multiprocessing_context": "spawn", "prefetch_factor": 1} if workers else {}),
    )
    totals = np.zeros((len(rows), len(classes)), dtype=np.float64)
    log_totals = np.full_like(totals, -np.inf)
    input_hash = _hash(
        {"data": data, "target": target, "inference": plan["inference"], "checkpoints": checkpoints}
    )
    cache_dir = folder / "members"
    ensure_managed_directory(cache_dir)
    try:
        for index, checkpoint in enumerate(checkpoints):
            if (folder / "cancel.requested").exists():
                raise KeyboardInterrupt("Evaluation cancelled.")
            snapshot = checkpoint_snapshot(checkpoint["path"], Path(checkpoint["path"]).parent)
            if any(snapshot[key] != checkpoint[key] for key in ("path", "bytes", "sha256")):
                raise ValueError("A frozen predictor checkpoint changed.")
            cache_path = cache_dir / f"member-{index}.json"
            probabilities = None
            log_probabilities = None
            if cache_path.exists():
                cache = json.loads(ScientificStore._read_file(cache_path, 64 * 1024 * 1024))
                if (
                    cache.get("inputHash") == input_hash
                    and cache.get("slideIds") == identities
                    and "logProbabilities" in cache
                    and cache.get("sha256")
                    == _hash(
                        {
                            "probabilities": cache.get("probabilities"),
                            "logProbabilities": cache.get("logProbabilities"),
                        }
                    )
                ):
                    probabilities = np.asarray(cache["probabilities"], dtype=np.float64)
                    log_probabilities = np.asarray(cache["logProbabilities"], dtype=np.float64)
            if probabilities is None:
                model = MILTrainModule.load_from_checkpoint(
                    checkpoint["path"], map_location="cpu", weights_only=True
                )
                if model.target != target or model.hparams.feature_dim != data["featureDim"]:
                    raise ValueError(
                        "The checkpoint target or feature dimensions differ from its predictor."
                    )
                model.eval().to(device)
                collected, log_collected, observed = [], [], []
                with torch.inference_mode():
                    for batch in loader:
                        if (folder / "cancel.requested").exists():
                            raise KeyboardInterrupt("Evaluation cancelled.")
                        with torch.autocast(
                            device_type=device.type,
                            dtype=torch.float16 if precision == "float16" else torch.bfloat16,
                            enabled=precision != "float32",
                        ):
                            logits = model(batch["features"].to(device), batch["mask"].to(device))
                        collected.extend(torch.softmax(logits.double(), dim=-1).cpu().tolist())
                        log_collected.extend(
                            torch.log_softmax(logits.double(), dim=-1).cpu().tolist()
                        )
                        observed.extend(batch["slideIds"])
                del model
                if observed != identities:
                    raise ValueError("Inference changed the selected slide order or membership.")
                probabilities = np.asarray(collected, dtype=np.float64)
                log_probabilities = np.asarray(log_collected, dtype=np.float64)
                write_json(
                    cache_path,
                    {
                        "inputHash": input_hash,
                        "slideIds": identities,
                        "probabilities": collected,
                        "logProbabilities": log_collected,
                        "sha256": _hash(
                            {"probabilities": collected, "logProbabilities": log_collected}
                        ),
                    },
                )
            if (
                probabilities.shape != totals.shape
                or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0)
                or np.any(probabilities > 1)
                or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)
                or log_probabilities.shape != totals.shape
                or not np.isfinite(log_probabilities).all()
                or not np.allclose(np.exp(log_probabilities), probabilities, atol=1e-6)
                or not np.allclose(np.logaddexp.reduce(log_probabilities, axis=1), 0, atol=1e-6)
            ):
                raise ValueError("A checkpoint produced invalid probabilities.")
            totals += probabilities / len(checkpoints)
            log_totals = np.logaddexp(log_totals, log_probabilities - np.log(len(checkpoints)))
            write_json(
                folder / "progress.json",
                {
                    "completedModels": index + 1,
                    "totalModels": len(checkpoints),
                    "slideCount": len(rows),
                },
            )
    finally:
        dataset.close()
    _check_inputs(data)
    records = [
        {
            "slideId": row["slideId"],
            "patientId": row.get("patientId"),
            **({"patientIdSource": row["patientIdSource"]} if "patientIdSource" in row else {}),
            "label": row.get("label"),
            "labelIndex": classes.index(row["label"]) if row.get("label") is not None else None,
            "probabilities": probability.tolist(),
            "logProbabilities": log_probability.tolist(),
        }
        for row, probability, log_probability in zip(rows, totals, log_totals, strict=True)
    ]
    threshold = plan["inference"]["decisionThreshold"]
    records = _decisions(records, target, threshold)
    try:
        patients = _decisions(patient_predictions(records), target, threshold)
        patient_metrics = evaluation_metrics(patients, target, threshold)
    except ValueError as error:
        if target["unit"] == "patient":
            raise
        patients, patient_metrics = [], {"available": False, "count": 0, "reason": str(error)}
    slide_metrics = evaluation_metrics(records, target, threshold)
    metrics = {
        "unit": target["unit"],
        "classOrder": classes,
        "positiveClass": target.get("positiveClass"),
        "decisionThreshold": threshold,
        "patientAggregation": "mean_probabilities",
        "slide": slide_metrics,
        "patient": patient_metrics,
        "selected": patient_metrics if target["unit"] == "patient" else slide_metrics,
    }
    write_json(
        folder / "predictions.json",
        {"classOrder": classes, "records": records, "patientRecords": patients},
    )
    write_json(folder / "metrics.json", metrics)
    _write_csv(folder / "slide-predictions.csv", records, classes)
    _write_csv(folder / "patient-predictions.csv", patients, classes, patient=True)
    artifacts = {}
    for name in (
        "predictions.json",
        "metrics.json",
        "slide-predictions.csv",
        "patient-predictions.csv",
    ):
        path = folder / name
        artifacts[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    result = {
        "state": "succeeded",
        "runId": plan.get("recordId", plan.get("runId")),
        "method": method,
        "checkpointCount": len(checkpoints),
        "slideCount": len(records),
        "patientCount": len(patients),
        "classOrder": classes,
        "metrics": metrics,
        "artifacts": artifacts,
        "inputHash": input_hash,
        "resumePolicy": "reuse_completed_members_replay_interrupted_member",
    }
    write_json(folder / "result.json", result)
    return result
