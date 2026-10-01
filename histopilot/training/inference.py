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

from histopilot.application.predictors import checkpoint_snapshot
from histopilot.cv_summary import evaluation_metrics
from histopilot.datasets.datamodule import _worker_init
from histopilot.datasets.mil import SlideDataset, collate_mil
from histopilot.inference_summary import describe, patient_member_probabilities, summarize
from histopilot.scoring import patient_predictions
from histopilot.storage.io import content_hash, read_file_bounded, write_json_atomic
from histopilot.storage.project_lock import ensure_managed_directory, reject_symlink_components
from histopilot.training.module import (
    MILTrainModule,
    class_logits,
    window_uncertainty_rows,
)
from histopilot.workers.train_batch import check_inputs


def _decisions(records, target, threshold):
    for row in records:
        if target["task"] == "binary_classification":
            positive = target["classes"].index(target["positiveClass"])
            predicted = positive if row["probabilities"][positive] >= threshold else 1 - positive
        else:
            predicted = int(np.argmax(row["probabilities"]))
        row.update(predictedIndex=predicted, predictedLabel=target["classes"][predicted])
    return records


# Member probabilities are recorded per record only while predictions.json stays
# well inside its 64 MiB artifact limit; larger ensembles omit them explicitly.
MAX_MEMBER_VALUES = 1_000_000
EVALUATION_COLUMNS = ("label", "predictedLabel")
# Inference exports carry no label column; they describe the decision instead.
INFERENCE_COLUMNS = ("predictedLabel", "confidence", "margin", "membersAgreeing", "memberCount")


def _write_csv(path, rows, classes, *, patient=False, columns=EVALUATION_COLUMNS):
    identity = ["patientId", "slideIds"] if patient else ["slideId", "patientId"]
    fields = identity + list(columns) + [f"probability:{label}" for label in classes]
    reject_symlink_components(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            value = {key: row.get(key) for key in fields[: len(identity) + len(columns)]}
            if patient:
                value["slideIds"] = json.dumps(row["slideIds"], ensure_ascii=False)
            value.update(
                {
                    f"probability:{label}": probability
                    for label, probability in zip(classes, row["probabilities"], strict=True)
                }
            )
            # Spreadsheet applications interpret these text cells as formulas,
            # even when CSV quoting is present. Preserve identifiers as text.
            writer.writerow(
                {
                    key: "'" + cell
                    if isinstance(cell, str)
                    and cell.lstrip().startswith(("=", "+", "-", "@", "\t", "\r"))
                    else cell
                    for key, cell in value.items()
                }
            )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _described_rows(rows, target, threshold):
    """Export rows describing each decision; predictions.json keeps only the evidence."""
    result = []
    for row in rows:
        described = describe(row, target, threshold)
        agreement = described.get("memberAgreement")
        result.append(
            {
                **row,
                "confidence": described["confidence"],
                "margin": described["margin"],
                "membersAgreeing": agreement["agree"] if agreement else None,
                "memberCount": agreement["total"] if agreement else None,
            }
        )
    return result


def _patient_members(records, patients, aggregation):
    """Transient patient member probabilities under the frozen slide-combination rule."""
    groups = {}
    for row in records:
        if row.get("patientId"):
            groups.setdefault(row["patientId"], []).append(row)
    result = []
    for patient in patients:
        members = patient_member_probabilities(groups[patient["patientId"]], aggregation)
        result.append({**patient, **({"memberProbabilities": members} if members else {})})
    return result


def _finish_inference(plan, folder, records, *, method, checkpoints, input_hash, member_evidence):
    """Predictions and a label-free summary; unlabeled rows are never scored."""
    target, classes = plan["target"], plan["target"]["classes"]
    threshold = plan["inference"]["decisionThreshold"]
    aggregation = plan["inference"]["patientAggregation"]
    slide_unit = plan.get("splitUnit") == "slide"
    if slide_unit:
        patients, patient_summary = (
            [],
            {
                "available": False,
                "count": 0,
                "reason": "Patient analysis is disabled for slide-level experiments.",
            },
        )
    else:
        try:
            patients = _decisions(patient_predictions(records, aggregation), target, threshold)
            patient_summary = summarize(
                _patient_members(records, patients, aggregation), target, threshold
            )
        except ValueError as error:
            if target["unit"] == "patient":
                raise
            patients, patient_summary = [], {"available": False, "count": 0, "reason": str(error)}
    slide_summary = summarize(records, target, threshold)
    # Label-blind evaluations share this output; their files still say what the run is.
    purpose = "evaluation" if plan.get("labelsWithheld") else "inference"
    summary = {
        "purpose": purpose,
        "unit": target["unit"],
        "classOrder": classes,
        "positiveClass": target.get("positiveClass"),
        "decisionThreshold": threshold,
        "patientAggregation": None
        if slide_unit
        else "mean_logits"
        if aggregation == "mean_logits"
        else "mean_probabilities",
        **({"splitUnit": plan["splitUnit"]} if "splitUnit" in plan else {}),
        "memberCount": len(checkpoints),
        "memberProbabilities": member_evidence,
        "slide": slide_summary,
        "patient": patient_summary,
        "selected": patient_summary if target["unit"] == "patient" else slide_summary,
    }
    write_json_atomic(
        folder / "predictions.json",
        {"classOrder": classes, "records": records, "patientRecords": patients},
    )
    write_json_atomic(folder / "summary.json", summary)
    _write_csv(
        folder / "slide-predictions.csv",
        _described_rows(records, target, threshold),
        classes,
        columns=INFERENCE_COLUMNS,
    )
    if not slide_unit:
        _write_csv(
            folder / "patient-predictions.csv",
            _described_rows(_patient_members(records, patients, aggregation), target, threshold),
            classes,
            patient=True,
            columns=INFERENCE_COLUMNS,
        )
    artifacts = {}
    for name in (
        "predictions.json",
        "summary.json",
        "slide-predictions.csv",
        *(() if slide_unit else ("patient-predictions.csv",)),
    ):
        path = folder / name
        artifacts[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    result = {
        "state": "succeeded",
        "purpose": purpose,
        **({"labelsWithheld": True} if plan.get("labelsWithheld") else {}),
        "runId": plan.get("recordId", plan.get("runId")),
        "method": method,
        "checkpointCount": len(checkpoints),
        "slideCount": len(records),
        "patientCount": len(patients),
        "classOrder": classes,
        "summary": summary,
        "artifacts": artifacts,
        "inputHash": input_hash,
        "resumePolicy": "reuse_completed_members_replay_interrupted_member",
    }
    write_json_atomic(folder / "result.json", result)
    return result


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
            or type(row["windowCount"]) is not int
            or row["windowCount"] < 1
        ):
            return False
        if any(
            isinstance(row[name], bool)
            or not isinstance(row[name], (int, float))
            or not np.isfinite(row[name])
            or not 0 <= row[name] <= np.log(shape[1]) + 1e-6
            for name in scalar_fields
        ):
            return False
        variance = row["probabilityVariance"]
        if (
            not isinstance(variance, list)
            or len(variance) != shape[1]
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or not 0 <= value <= 0.25 + 1e-6
                for value in variance
            )
        ):
            return False
    return True


def _cached_member(path, input_hash, member_hash, identities, shape, *, include_uncertainty=False):
    if not path.exists() and not path.is_symlink():
        return None
    # Filesystem safety failures remain errors; malformed derived predictions can
    # be discarded and recomputed from the still-verified checkpoint and features.
    content = read_file_bounded(path, 64 * 1024 * 1024)
    try:
        cache = json.loads(content)
        if (
            not isinstance(cache, dict)
            or cache.get("inputHash") != input_hash
            or cache.get("memberHash") != member_hash
            or cache.get("slideIds") != identities
            or "logProbabilities" not in cache
            or cache.get("sha256")
            != content_hash(
                {
                    "probabilities": cache.get("probabilities"),
                    "logProbabilities": cache.get("logProbabilities"),
                    **(
                        {"windowUncertainty": cache["windowUncertainty"]}
                        if "windowUncertainty" in cache
                        else {}
                    ),
                }
            )
        ):
            return None
        probabilities = np.asarray(cache["probabilities"], dtype=np.float64)
        log_probabilities = np.asarray(cache["logProbabilities"], dtype=np.float64)
        uncertainty = cache.get("windowUncertainty")
        if _valid_probabilities(probabilities, log_probabilities, shape) and (
            uncertainty is None or _valid_window_uncertainty(uncertainty, shape)
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
        method not in {"ensemble", "refit", "seed_ensemble"}
        or not checkpoints
        or (method == "refit" and len(checkpoints) != 1)
        or (method == "seed_ensemble" and len(checkpoints) < 2)
    ):
        raise ValueError(
            "A refit requires one checkpoint; an ensemble requires its selected fold checkpoints."
        )
    # A seed ensemble pools fold models trained under different training seeds. Each keeps
    # its own seed for evaluation-bag subsampling, as its per-seed ensemble would.
    if method == "seed_ensemble" and any(
        type(item.get("trainingSeed")) is not int or item["trainingSeed"] < 0
        for item in checkpoints
    ):
        raise ValueError("Every seed-ensemble member must record its training seed.")
    purpose = plan.get("purpose")
    if purpose not in {None, "inference"}:
        raise ValueError("Unsupported evaluation purpose.")
    # A labeled evaluation saved for service scoring withholds its labels from this job:
    # it predicts exactly as inference does, and the control service scores the result.
    labels_withheld = plan.get("labelsWithheld", False)
    if type(labels_withheld) is not bool or (labels_withheld and purpose is not None):
        raise ValueError("Only a labeled evaluation can withhold its labels.")
    inference_only = purpose == "inference" or labels_withheld
    split_unit = plan.get("splitUnit", "patient")
    if split_unit not in {"slide", "patient"}:
        raise ValueError("Unsupported split unit.")
    slide_unit = split_unit == "slide"
    if "splitUnit" in plan and target["unit"] != split_unit:
        raise ValueError("The prediction target must match the frozen split unit.")
    rows = data["memberships"]
    identities = [row["slideId"] for row in rows]
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("Inference requires exactly one membership per selected slide.")
    if inference_only and any(row.get("label") is not None for row in rows):
        raise ValueError(
            "Label-blind evaluations predict unlabeled memberships only."
            if labels_withheld
            else "Inference runs predict unlabeled memberships only."
        )
    if any(row.get("label") is not None and row["label"] not in classes for row in rows):
        raise ValueError("Test labels must preserve the frozen target class order.")
    if target["unit"] == "patient" and any(not row.get("patientId") for row in rows):
        raise ValueError("Patient evaluation requires a grouping identity for every slide.")
    patient_aggregation = plan["inference"]["patientAggregation"]
    if not slide_unit and patient_aggregation not in {"mean", "mean_logits"}:
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
    check_inputs(data)
    memberships = [
        {**row, "labelIndex": classes.index(row["label"]) if row.get("label") is not None else -1}
        for row in rows
    ]
    dataset = SlideDataset(
        {
            **data,
            "target": target,
            **({"splitUnit": plan["splitUnit"]} if "splitUnit" in plan else {}),
            "trainingSeed": plan.get("bagPolicy", {}).get("trainingSeed", 0),
            "recipe": {
                "bagSize": None,
                "evalBagSize": plan.get("bagPolicy", {}).get("evalBagSize"),
                "inputMode": data.get("inputMode", "image"),
                "clinicalFields": data.get("clinicalFields", []),
            },
        },
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
    # Decide before collecting vectors: a late artifact-size check still retains
    # every ensemble member in RAM, including for ordinary evaluation jobs.
    member_evidence = "single_model" if len(checkpoints) == 1 else "recorded"
    # Log probabilities are essential when patient voting averages logits:
    # exponentiation may round an extreme but finite probability down to zero.
    retain_member_logs = not slide_unit and patient_aggregation == "mean_logits"
    evidence_width = 2 if retain_member_logs else 1
    if len(rows) * len(checkpoints) * len(classes) * evidence_width > MAX_MEMBER_VALUES:
        member_evidence = "omitted_for_size" if len(checkpoints) > 1 else "single_model"
    member_probabilities = (
        [[] for _ in rows] if inference_only and member_evidence == "recorded" else None
    )
    member_logs = (
        [[] for _ in rows] if member_probabilities is not None and retain_member_logs else None
    )
    input_hash = content_hash(
        {
            "data": data,
            "target": target,
            **({"splitUnit": plan["splitUnit"]} if "splitUnit" in plan else {}),
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
            member_hash = content_hash({"inputHash": input_hash, "checkpoint": checkpoint})
            cached = _cached_member(
                cache_path,
                input_hash,
                member_hash,
                identities,
                totals.shape,
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
                if model.recipe.get("inputMode", "image") != data.get(
                    "inputMode", "image"
                ) or model.recipe.get("clinicalFields", []) != data.get("clinicalFields", []):
                    raise ValueError(
                        "Checkpoint clinical schema differs from its frozen evaluation."
                    )
                model.eval().to(device)
                if "trainingSeed" in checkpoint:
                    # Loader workers copy the dataset when each member's pass starts.
                    dataset.training_seed = checkpoint["trainingSeed"]
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
                                batch["features"].to(device),
                                batch["mask"].to(device),
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
                            uncertainty_collected.extend(
                                window_uncertainty_rows(output["window_uncertainty"], target)
                            )
                del model
                if observed != identities:
                    raise ValueError("Inference changed the selected slide order or membership.")
                probabilities = np.asarray(collected, dtype=np.float64)
                log_probabilities = np.asarray(log_collected, dtype=np.float64)
                if not _valid_probabilities(probabilities, log_probabilities, totals.shape):
                    raise ValueError("A checkpoint produced invalid probabilities.")
                uncertainty = uncertainty_collected or None
                if uncertainty is not None and not _valid_window_uncertainty(
                    uncertainty, totals.shape
                ):
                    raise ValueError("A checkpoint produced invalid feature-window uncertainty.")
                payload = {
                    "probabilities": collected,
                    "logProbabilities": log_collected,
                    **({"windowUncertainty": uncertainty} if uncertainty is not None else {}),
                }
                write_json_atomic(
                    cache_path,
                    {
                        "inputHash": input_hash,
                        "memberHash": member_hash,
                        "slideIds": identities,
                        **payload,
                        "sha256": content_hash(payload),
                    },
                )
            else:
                probabilities, log_probabilities, uncertainty = cached
            if uncertainty is not None:
                for members, scores in zip(window_uncertainty_members, uncertainty, strict=True):
                    members.append(
                        {
                            "memberIndex": index,
                            "checkpointSha256": checkpoint["sha256"],
                            **scores,
                        }
                    )
            if member_probabilities is not None:
                for members, values in zip(
                    member_probabilities, probabilities.tolist(), strict=True
                ):
                    members.append(values)
            if member_logs is not None:
                for members, values in zip(member_logs, log_probabilities.tolist(), strict=True):
                    members.append(values)
            totals += probabilities / len(checkpoints)
            log_totals = np.logaddexp(log_totals, log_probabilities - np.log(len(checkpoints)))
            logit_totals += log_probabilities / len(checkpoints)
            write_json_atomic(
                folder / "progress.json",
                {
                    "completedModels": index + 1,
                    "totalModels": len(checkpoints),
                    "slideCount": len(rows),
                },
            )
    finally:
        dataset.close()
    check_inputs(data)
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
    if member_probabilities is not None:
        # Each fold member's own probabilities expose ensemble agreement without labels.
        for row, values in zip(records, member_probabilities, strict=True):
            row["memberProbabilities"] = values
    if member_logs is not None:
        for row, values in zip(records, member_logs, strict=True):
            row["memberLogProbabilities"] = values
    threshold = plan["inference"]["decisionThreshold"]
    records = _decisions(records, target, threshold)
    if inference_only:
        return _finish_inference(
            plan,
            folder,
            records,
            method=method,
            checkpoints=checkpoints,
            input_hash=input_hash,
            member_evidence=member_evidence,
        )
    if slide_unit:
        patients, patient_metrics = (
            [],
            {
                "available": False,
                "count": 0,
                "reason": "Patient analysis is disabled for slide-level experiments.",
            },
        )
    else:
        try:
            patients = _decisions(
                patient_predictions(records, patient_aggregation), target, threshold
            )
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
        "patientAggregation": None
        if slide_unit
        else "mean_logits"
        if patient_aggregation == "mean_logits"
        else "mean_probabilities",
        **({"splitUnit": plan["splitUnit"]} if "splitUnit" in plan else {}),
        **({"ensembleAggregation": aggregation} if aggregation == "mean_logit" else {}),
        "slide": slide_metrics,
        "patient": patient_metrics,
        "selected": patient_metrics if target["unit"] == "patient" else slide_metrics,
    }
    if not slide_unit and plan.get("analysis") is not None:
        from histopilot.statistics import patient_analysis

        metrics["patientAnalysis"] = patient_analysis(records, patients, target, plan["analysis"])
        intervals = metrics["patientAnalysis"]["uncertainty"].get("intervals")
        if intervals:
            patient_metrics["confidenceIntervals"] = intervals
    write_json_atomic(
        folder / "predictions.json",
        {"classOrder": classes, "records": records, "patientRecords": patients},
    )
    write_json_atomic(folder / "metrics.json", metrics)
    _write_csv(folder / "slide-predictions.csv", records, classes)
    if not slide_unit:
        _write_csv(folder / "patient-predictions.csv", patients, classes, patient=True)
    artifacts = {}
    for name in (
        "predictions.json",
        "metrics.json",
        "slide-predictions.csv",
        *(() if slide_unit else ("patient-predictions.csv",)),
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
    write_json_atomic(folder / "result.json", result)
    return result
