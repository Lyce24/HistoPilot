"""Frozen slide and patient inference with recorded bag and ensemble policies."""

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from histopilot.application.feature_bundles import _hash
from histopilot.application.predictors import checkpoint_snapshot
from histopilot.datasets.datamodule import _worker_init
from histopilot.datasets.mil import SlideDataset, collate_mil
from histopilot.scoring import patient_predictions
from histopilot.storage.project_lock import _reject_symlink_components, ensure_managed_directory
from histopilot.storage.scientific import ScientificStore
from histopilot.training.module import (
    MILTrainModule,
    _metrics,
    class_logits,
    window_uncertainty_rows,
)
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
    result = _metrics(labeled, target, decision_threshold=threshold)
    return {
        **result,
        "predictionCount": len(records),
        "unlabeledCount": len(records) - len(labeled),
    }


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


def _valid_probabilities(probabilities, log_probabilities, shape):
    return (
        probabilities.shape == shape
        and log_probabilities.shape == shape
        and np.isfinite(probabilities).all()
        and np.all(probabilities >= 0)
        and np.all(probabilities <= 1)
        and np.allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=0)
        and np.isfinite(log_probabilities).all()
        and np.allclose(np.exp(log_probabilities), probabilities, atol=1e-6, rtol=0)
        and np.allclose(np.logaddexp.reduce(log_probabilities, axis=1), 0, atol=1e-6, rtol=0)
    )


def _valid_window_uncertainty(rows, shape):
    if not isinstance(rows, list) or len(rows) != shape[0]:
        return False
    scalar_fields = ("entropyOfMeanProbability", "meanWindowEntropy", "mutualInformation")
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != {"windowCount", "probabilityVariance", *scalar_fields}
            or type(row["windowCount"]) is not int or row["windowCount"] < 1
        ):
            return False
        if any(
            isinstance(row[name], bool) or not isinstance(row[name], (int, float))
            or not np.isfinite(row[name]) or not 0 <= row[name] <= np.log(shape[1]) + 1e-6
            for name in scalar_fields
        ):
            return False
        variance = row["probabilityVariance"]
        if (
            not isinstance(variance, list) or len(variance) != shape[1]
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not np.isfinite(value) or not 0 <= value <= 0.25 + 1e-6 for value in variance)
        ):
            return False
    return True


def _cached_member(path, input_hash, member_hash, identities, shape, *, include_uncertainty=False):
    if not path.exists() and not path.is_symlink():
        return None
    # Filesystem safety failures remain errors; malformed derived predictions can
    # be discarded and recomputed from the still-verified checkpoint and features.
    content = ScientificStore._read_file(path, 64 * 1024 * 1024)
    try:
        cache = json.loads(content)
        if (
            not isinstance(cache, dict)
            or cache.get("inputHash") != input_hash
            or cache.get("memberHash") != member_hash
            or cache.get("slideIds") != identities
            or "logProbabilities" not in cache
            or cache.get("sha256")
            != _hash(
                {
                    "probabilities": cache.get("probabilities"),
                    "logProbabilities": cache.get("logProbabilities"),
                    **({"windowUncertainty": cache["windowUncertainty"]}
                       if "windowUncertainty" in cache else {}),
                }
            )
        ):
            return None
        probabilities = np.asarray(cache["probabilities"], dtype=np.float64)
        log_probabilities = np.asarray(cache["logProbabilities"], dtype=np.float64)
        uncertainty = cache.get("windowUncertainty")
        if (
            _valid_probabilities(probabilities, log_probabilities, shape)
            and (uncertainty is None or _valid_window_uncertainty(uncertainty, shape))
        ):
            if include_uncertainty:
                return probabilities, log_probabilities, uncertainty
            return probabilities, log_probabilities
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        pass
    return None


def evaluate(plan, output_dir):
    """Run each model sequentially so ensembles do not multiply GPU model memory.

    Completed member predictions are cached with a digest, the full input
    fingerprint, and the individual checkpoint identity. Resume restarts the
    interrupted member and reuses verified completed ones.
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
    patient_aggregation = plan["inference"]["patientAggregation"]
    if patient_aggregation not in {"mean", "mean_logits"}:
        raise ValueError("Frozen predictors require mean probabilities or mean logits.")
    aggregation = plan.get("aggregation", "mean_probability")
    if aggregation not in {"mean_probability", "mean_logit", "single_model"}:
        raise ValueError("The frozen ensemble aggregation is unsupported.")
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
        {**data, "target": target,
         "trainingSeed": plan.get("bagPolicy", {}).get("trainingSeed", 0),
         "recipe": {"bagSize": None, "evalBagSize": plan.get("bagPolicy", {}).get("evalBagSize"),
                    "inputMode": data.get("inputMode", "image"),
                    "clinicalFields": data.get("clinicalFields", [])}},
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
    logit_totals = np.zeros_like(totals)
    window_uncertainty_members = [[] for _ in rows]
    input_hash = _hash(
        {
            "data": data,
            "target": target,
            "inference": plan["inference"],
            "checkpoints": checkpoints,
            **({"bagPolicy": plan["bagPolicy"]} if "bagPolicy" in plan else {}),
            **({"aggregation": aggregation} if aggregation == "mean_logit" else {}),
        }
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
            member_hash = _hash({"inputHash": input_hash, "checkpoint": checkpoint})
            cached = _cached_member(
                cache_path, input_hash, member_hash, identities, totals.shape,
                include_uncertainty=True,
            )
            if cached is None:
                model = MILTrainModule.load_from_checkpoint(
                    checkpoint["path"], map_location="cpu", weights_only=True
                )
                if model.target != target or model.hparams.feature_dim != data["featureDim"]:
                    raise ValueError(
                        "The checkpoint target or feature dimensions differ from its predictor."
                    )
                if (model.recipe.get("inputMode", "image") != data.get("inputMode", "image")
                        or model.recipe.get("clinicalFields", []) != data.get("clinicalFields", [])):
                    raise ValueError("Checkpoint clinical schema differs from its frozen evaluation.")
                model.eval().to(device)
                collected, log_collected, observed, uncertainty_collected = [], [], [], []
                with torch.inference_mode():
                    for batch in loader:
                        if (folder / "cancel.requested").exists():
                            raise KeyboardInterrupt("Evaluation cancelled.")
                        with torch.autocast(
                            device_type=device.type,
                            dtype=torch.float16 if precision == "float16" else torch.bfloat16,
                            enabled=precision != "float32",
                        ):
                            output = model.prediction_output(
                                batch["features"].to(device), batch["mask"].to(device),
                                **({"clinical": batch["clinical"]} if "clinical" in batch else {}),
                            )
                            logits = output["logits"]
                        logits = class_logits(logits, target)
                        collected.extend(torch.softmax(logits.double(), dim=-1).cpu().tolist())
                        log_collected.extend(
                            torch.log_softmax(logits.double(), dim=-1).cpu().tolist()
                        )
                        observed.extend(batch["slideIds"])
                        if output.get("window_uncertainty") is not None:
                            uncertainty_collected.extend(window_uncertainty_rows(
                                output["window_uncertainty"], target
                            ))
                del model
                if observed != identities:
                    raise ValueError("Inference changed the selected slide order or membership.")
                probabilities = np.asarray(collected, dtype=np.float64)
                log_probabilities = np.asarray(log_collected, dtype=np.float64)
                if not _valid_probabilities(probabilities, log_probabilities, totals.shape):
                    raise ValueError("A checkpoint produced invalid probabilities.")
                uncertainty = uncertainty_collected or None
                if uncertainty is not None and not _valid_window_uncertainty(uncertainty, totals.shape):
                    raise ValueError("A checkpoint produced invalid feature-window uncertainty.")
                payload = {
                    "probabilities": collected, "logProbabilities": log_collected,
                    **({"windowUncertainty": uncertainty} if uncertainty is not None else {}),
                }
                write_json(
                    cache_path,
                    {
                        "inputHash": input_hash,
                        "memberHash": member_hash,
                        "slideIds": identities,
                        **payload,
                        "sha256": _hash(payload),
                    },
                )
            else:
                probabilities, log_probabilities, uncertainty = cached
            if uncertainty is not None:
                for members, scores in zip(window_uncertainty_members, uncertainty, strict=True):
                    members.append({
                        "memberIndex": index, "checkpointSha256": checkpoint["sha256"], **scores,
                    })
            totals += probabilities / len(checkpoints)
            log_totals = np.logaddexp(log_totals, log_probabilities - np.log(len(checkpoints)))
            logit_totals += log_probabilities / len(checkpoints)
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
    if aggregation == "mean_logit":
        log_totals = logit_totals - np.logaddexp.reduce(logit_totals, axis=1, keepdims=True)
        totals = np.exp(log_totals)
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
    for row, members in zip(records, window_uncertainty_members, strict=True):
        if members:
            row["windowUncertaintyByMember"] = members
    threshold = plan["inference"]["decisionThreshold"]
    records = _decisions(records, target, threshold)
    try:
        patients = _decisions(patient_predictions(records, patient_aggregation), target, threshold)
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
        "patientAggregation": "mean_logits"
        if patient_aggregation == "mean_logits"
        else "mean_probabilities",
        **({"ensembleAggregation": aggregation} if aggregation == "mean_logit" else {}),
        "slide": slide_metrics,
        "patient": patient_metrics,
        "selected": patient_metrics if target["unit"] == "patient" else slide_metrics,
    }
    if plan.get("analysis") is not None:
        from histopilot.statistics import patient_analysis

        metrics["patientAnalysis"] = patient_analysis(
            records, patients, target, plan["analysis"]
        )
        intervals = metrics["patientAnalysis"]["uncertainty"].get("intervals")
        if intervals:
            patient_metrics["confidenceIntervals"] = intervals
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
