"""Whole-bag attention maps of frozen refits and ensembles, for every attention model."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from histopilot.application.predictors import checkpoint_snapshot
from histopilot.models import catalog
from histopilot.storage.attention_inputs import inspect_inputs, verify_sources
from histopilot.storage.io import content_hash, read_file_bounded, write_json_atomic
from histopilot.storage.packed import open_source
from histopilot.storage.project_lock import ensure_managed_directory, reject_symlink_components
from histopilot.training.module import MILTrainModule, class_logits, window_uncertainty_rows

ATTENTION_NOTE = (
    "Class-independent pooling weights, normalized over all patches in this slide. "
    "For nnMIL, each member averages normalized attention across its feature windows. "
    "The ensemble mean averages each member's attention. Percentiles are "
    "within-slide midranks for display, not calibrated probabilities or class-specific attribution."
)
POOLED_ATTENTION_NOTE = (
    "Pooled attention: each selected model's normalized map (an ensemble's maps averaged across "
    "its members) is combined patch by patch with the chosen function, then renormalized over "
    "all patches in this slide. The pooled prediction is the equal-weight mean of the models' "
    "class probabilities. Percentiles are within-slide midranks for display, not calibrated "
    "probabilities or class-specific attribution."
)
# Each function maps a (models × patches) stack of normalized maps to one nonnegative map.
POOLING_FUNCTIONS = {
    "mean": lambda maps: maps.mean(axis=0),
    "median": lambda maps: np.median(maps, axis=0),
    "max": lambda maps: maps.max(axis=0),
    "min": lambda maps: maps.min(axis=0),
    # Consensus: a patch ranks high only where every model attends to it.
    "geometric_mean": lambda maps: np.exp(np.log(np.clip(maps, 1e-300, None)).mean(axis=0)),
}
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024


def attention_note(model):
    """How one architecture's member maps are formed; ABMIL and nnMIL keep ATTENTION_NOTE."""
    found = catalog.spec(model)
    if found is None or found.attention_note is None or found.name == "nnmil":
        return ATTENTION_NOTE
    return (
        f"Per-patch attention weights, normalized over all patches in this slide. "
        f"{found.attention_note} The ensemble mean averages each member's attention. "
        "Percentiles are within-slide midranks for display, not calibrated probabilities."
    )


def _write_attention_json(path, value):
    """Stream large maps atomically instead of using the 64-MiB metadata writer."""
    reject_symlink_components(path)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    size = 0
    try:
        with os.fdopen(descriptor, "wb") as stream:
            encoder = json.JSONEncoder(
                sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            )
            for part in encoder.iterencode(value):
                encoded = part.encode("utf-8")
                size += len(encoded)
                if size > MAX_ARTIFACT_BYTES:
                    raise ValueError("Attention output exceeds its 512-MiB artifact budget.")
                stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_attention_array(path, coords, weights):
    """A fixed N×4 float64 array supports bounded, memory-mapped viewport reads."""
    reject_symlink_components(path)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            array = np.empty((len(coords), 4), dtype="<f8")
            array[:, :2], array[:, 2], array[:, 3] = coords, weights, _percentiles(weights)
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _receipt(path):
    with open_source(path) as (stream, stamp):
        if not 0 < stamp["sizeBytes"] <= MAX_ARTIFACT_BYTES:
            raise ValueError("Attention artifact is empty or exceeds its byte budget.")
        digest = hashlib.sha256()
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return {"path": str(path), "bytes": stamp["sizeBytes"], "sha256": digest.hexdigest()}


def _percentiles(weights):
    _, inverse, counts = np.unique(weights, return_inverse=True, return_counts=True)
    return ((np.cumsum(counts) - counts / 2) / len(weights))[inverse]


def _map(
    slide,
    coords,
    weights,
    probabilities,
    plan,
    *,
    member,
    log_probabilities=None,
    window_uncertainty=None,
    note=ATTENTION_NOTE,
):
    ranks = _percentiles(weights)
    return {
        "slideId": slide["slideId"],
        "patchCount": len(weights),
        "patchWidthLevel0": slide["patchWidthLevel0"],
        "patchHeightLevel0": slide["patchHeightLevel0"],
        "width": slide["width"],
        "height": slide["height"],
        "coordinateSpace": "level0",
        "attentionKind": "class_independent_pooling",
        "attentionNote": note,
        "classOrder": plan["target"]["classes"],
        "member": member,
        "probabilities": probabilities.tolist(),
        **({"windowUncertainty": window_uncertainty} if window_uncertainty is not None else {}),
        **(
            {"logProbabilities": log_probabilities.tolist()}
            if log_probabilities is not None
            else {}
        ),
        "patches": [
            {
                "index": index,
                "x": int(xy[0]),
                "y": int(xy[1]),
                "weight": float(weight),
                "percentile": float(rank),
            }
            for index, (xy, weight, rank) in enumerate(zip(coords, weights, ranks, strict=True))
        ],
    }


def _validate(weights, probabilities, patch_count, classes):
    if (
        weights.shape != (patch_count,)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1, atol=1e-6)
        or probabilities.shape != (classes,)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.isclose(probabilities.sum(), 1, atol=1e-6)
    ):
        raise ValueError("MIL produced invalid attention or class probabilities.")


def _validate_log_probabilities(log_probabilities, probabilities):
    if (
        log_probabilities.shape != probabilities.shape
        or not np.isfinite(log_probabilities).all()
        or not np.isclose(np.logaddexp.reduce(log_probabilities), 0, atol=1e-6)
        or not np.allclose(np.exp(log_probabilities), probabilities, atol=1e-6, rtol=1e-5)
    ):
        raise ValueError(
            "Saved attention log probabilities must be finite and match probabilities."
        )


def interpret(plan, output_dir):
    """Resume completed slide/member pairs; interrupted pairs recompute the whole bag.

    Every attention value comes from the same complete-bag forward call as its
    member's class probabilities. There is no top-k filtering or local softmax.
    """
    folder = Path(output_dir)
    ensure_managed_directory(folder)
    if plan["method"] == "pooled":
        return _interpret_pooled(plan, folder)
    method, checkpoints = plan["method"], plan["checkpoints"]
    aggregation = plan.get("aggregation", "mean_probability")
    if aggregation not in {"mean_probability", "mean_logit", "single_model"}:
        raise ValueError("The frozen ensemble aggregation is unsupported.")
    if (
        method not in {"refit", "ensemble", "seed_ensemble"}
        or not checkpoints
        or (method == "refit" and len(checkpoints) != 1)
    ):
        raise ValueError(
            "Refit attention requires one checkpoint; ensembles require every frozen member."
        )
    if not plan["slides"] or len({row["slideId"] for row in plan["slides"]}) != len(plan["slides"]):
        raise ValueError("Select distinct nonempty slide identities.")
    verify_sources(plan["slides"])
    for checkpoint in checkpoints:
        current = checkpoint_snapshot(checkpoint["path"], Path(checkpoint["path"]).parent)
        if any(current[key] != checkpoint[key] for key in ("path", "sha256", "bytes")):
            raise ValueError("A frozen predictor checkpoint changed.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(plan["resources"]["cpuThreadsPerRun"])
    torch.use_deterministic_algorithms(True)
    device = torch.device(plan.get("device", "cpu"))
    input_contract = {
        key: plan.get(key)
        for key in ("method", "target", "featureContract", "slides", "checkpoints", "code")
    }
    if aggregation == "mean_logit":
        input_contract["aggregation"] = aggregation
    input_hash = content_hash(input_contract)
    artifacts, summaries = {}, []
    # Every member of one predictor shares its architecture, and so its note.
    run_note = ATTENTION_NOTE
    for slide_index, slide in enumerate(plan["slides"]):
        if (folder / "cancel.requested").exists():
            raise KeyboardInterrupt("Interpretation cancelled.")
        evidence, features, coords = inspect_inputs(
            slide, slide, plan["featureContract"], load=True
        )
        if any(slide.get(key) != value for key, value in evidence.items()):
            raise ValueError(
                "Feature vectors, coordinates, or geometry differ from reviewed evidence."
            )
        tensor = torch.from_numpy(features).unsqueeze(0).to(device)
        total_weights = np.zeros(len(coords), dtype=np.float64)
        total_probabilities = np.zeros(len(plan["target"]["classes"]), dtype=np.float64)
        total_log_probabilities = np.zeros_like(total_probabilities)
        members = []
        for member_index, checkpoint in enumerate(checkpoints):
            if (folder / "cancel.requested").exists():
                raise KeyboardInterrupt("Interpretation cancelled.")
            filename = f"slide-{slide_index}-member-{member_index}.json"
            path = folder / filename
            cache_receipt = folder / f".slide-{slide_index}-member-{member_index}.receipt.json"
            weights, probabilities, log_probabilities = None, None, None
            window_uncertainty = None
            if cache_receipt.exists() and path.exists():
                cached = json.loads(read_file_bounded(cache_receipt, 4096))
                if cached.get("inputHash") != input_hash or cached.get("artifact") != _receipt(
                    path
                ):
                    raise ValueError(
                        "Saved attention member evidence changed; it cannot be resumed."
                    )
                value = json.loads(read_file_bounded(path, MAX_ARTIFACT_BYTES))
                if (
                    value.get("slideId") != slide["slideId"]
                    or value.get("member") != str(member_index)
                    or value.get("classOrder") != plan["target"]["classes"]
                ):
                    raise ValueError("Saved attention member has a different slide or predictor.")
                cached_coords = np.asarray([[patch["x"], patch["y"]] for patch in value["patches"]])
                if not np.array_equal(coords, cached_coords) or [
                    patch["index"] for patch in value["patches"]
                ] != list(range(len(coords))):
                    raise ValueError("Saved attention coordinates changed row order.")
                weights = np.asarray(
                    [patch["weight"] for patch in value["patches"]], dtype=np.float64
                )
                probabilities = np.asarray(value["probabilities"], dtype=np.float64)
                window_uncertainty = value.get("windowUncertainty")
                run_note = value.get("attentionNote", ATTENTION_NOTE)
                if aggregation == "mean_logit":
                    log_probabilities = np.asarray(value.get("logProbabilities"), dtype=np.float64)
                    _validate_log_probabilities(log_probabilities, probabilities)
            if weights is None:
                model = MILTrainModule.load_from_checkpoint(
                    checkpoint["path"], map_location="cpu", weights_only=True
                )
                if (
                    model.target != plan["target"]
                    or model.hparams.feature_dim != plan["featureContract"]["dimensions"]
                    or not catalog.supports_attention(
                        model.recipe.get("model"), model.recipe.get("inputMode", "image")
                    )
                ):
                    raise ValueError(
                        "Checkpoint architecture, target or feature dimensions differ from the predictor."
                    )
                run_note = attention_note(model.recipe.get("model"))
                model.eval().to(device)
                with torch.inference_mode():
                    output = model.prediction_output(
                        tensor,
                        return_attention=True,
                        **({"clinical": [slide["clinical"]]} if "clinical" in slide else {}),
                    )
                    weights = output["attention"][0].double().cpu().numpy()
                    logits = class_logits(output["logits"], model.target)[0].double()
                    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
                    if aggregation == "mean_logit":
                        log_probabilities = torch.log_softmax(logits, dim=-1).cpu().numpy()
                    if output.get("window_uncertainty") is not None:
                        window_uncertainty = window_uncertainty_rows(
                            output["window_uncertainty"], model.target
                        )[0]
                del output, model
                # Double-precision renormalization removes softmax roundoff only.
                weights = weights / weights.sum()
                _validate(weights, probabilities, len(coords), len(total_probabilities))
                _write_attention_json(
                    path,
                    _map(
                        slide,
                        coords,
                        weights,
                        probabilities,
                        plan,
                        member=str(member_index),
                        log_probabilities=log_probabilities,
                        window_uncertainty=window_uncertainty,
                        note=run_note,
                    ),
                )
                write_json_atomic(
                    cache_receipt, {"inputHash": input_hash, "artifact": _receipt(path)}
                )
            _validate(weights, probabilities, len(coords), len(total_probabilities))
            artifacts[filename] = _receipt(path)
            array_name = f"slide-{slide_index}-member-{member_index}.npy"
            _write_attention_array(folder / array_name, coords, weights)
            artifacts[array_name] = _receipt(folder / array_name)
            total_weights += weights / len(checkpoints)
            total_probabilities += probabilities / len(checkpoints)
            if aggregation == "mean_logit":
                _validate_log_probabilities(log_probabilities, probabilities)
                total_log_probabilities += log_probabilities / len(checkpoints)
            members.append(
                {
                    "index": member_index,
                    "checkpointSha256": checkpoint["sha256"],
                    "probabilities": probabilities.tolist(),
                    "attentionArtifact": filename,
                    "attentionArray": array_name,
                    **(
                        {"windowUncertainty": window_uncertainty}
                        if window_uncertainty is not None
                        else {}
                    ),
                }
            )
            write_json_atomic(
                folder / "progress.json",
                {
                    "completedPairs": slide_index * len(checkpoints) + member_index + 1,
                    "totalPairs": len(plan["slides"]) * len(checkpoints),
                    "currentSlide": slide["slideId"],
                    "completedSlides": slide_index,
                    "totalSlides": len(plan["slides"]),
                    "completedModels": member_index + 1,
                    "totalModels": len(checkpoints),
                },
            )
        del tensor, features
        if aggregation == "mean_logit":
            total_log_probabilities -= np.logaddexp.reduce(total_log_probabilities)
            total_probabilities = np.exp(total_log_probabilities)
        _validate(total_weights, total_probabilities, len(coords), len(total_probabilities))
        filename = f"slide-{slide_index}.json"
        _write_attention_json(
            folder / filename,
            _map(
                slide,
                coords,
                total_weights,
                total_probabilities,
                plan,
                member="mean",
                log_probabilities=total_log_probabilities if aggregation == "mean_logit" else None,
                note=run_note,
            ),
        )
        artifacts[filename] = _receipt(folder / filename)
        array_name = f"slide-{slide_index}.npy"
        _write_attention_array(folder / array_name, coords, total_weights)
        artifacts[array_name] = _receipt(folder / array_name)
        summaries.append(
            {
                "slideId": slide["slideId"],
                "patchCount": len(coords),
                "probabilities": total_probabilities.tolist(),
                "attentionArtifact": filename,
                "attentionArray": array_name,
                "members": members,
            }
        )
    verify_sources(plan["slides"])
    write_json_atomic(
        folder / "attention.json",
        {
            "runId": plan["runId"],
            "method": method,
            "classOrder": plan["target"]["classes"],
            "attentionKind": "class_independent_pooling",
            "attentionNote": run_note,
            **({"ensembleAggregation": aggregation} if aggregation == "mean_logit" else {}),
            "slides": summaries,
        },
    )
    artifacts["attention.json"] = _receipt(folder / "attention.json")
    return {
        "state": "succeeded",
        "runId": plan["runId"],
        "method": method,
        "slideCount": len(summaries),
        "memberCount": len(checkpoints),
        "classOrder": plan["target"]["classes"],
        "attentionKind": "class_independent_pooling",
        "attentionNote": run_note,
        **({"ensembleAggregation": aggregation} if aggregation == "mean_logit" else {}),
        "slides": summaries,
        "artifacts": artifacts,
    }


def _checkpoint_attention(checkpoint, plan, tensor, slide, device, mean_logit):
    """One checkpoint's whole-bag attention and class probabilities, as in interpret(), and
    the note that says how its architecture's map is formed."""
    model = MILTrainModule.load_from_checkpoint(
        checkpoint["path"], map_location="cpu", weights_only=True
    )
    if (
        model.target != plan["target"]
        or model.hparams.feature_dim != plan["featureContract"]["dimensions"]
        or not catalog.supports_attention(
            model.recipe.get("model"), model.recipe.get("inputMode", "image")
        )
    ):
        raise ValueError(
            "Checkpoint architecture, target or feature dimensions differ from the predictor."
        )
    model.eval().to(device)
    with torch.inference_mode():
        output = model.prediction_output(
            tensor,
            return_attention=True,
            **({"clinical": [slide["clinical"]]} if "clinical" in slide else {}),
        )
        weights = output["attention"][0].double().cpu().numpy()
        logits = class_logits(output["logits"], model.target)[0].double()
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        log_probabilities = torch.log_softmax(logits, dim=-1).cpu().numpy() if mean_logit else None
    note = attention_note(model.recipe.get("model"))
    del output, model
    return weights / weights.sum(), probabilities, log_probabilities, note


def _interpret_pooled(plan, folder):
    """Several models' maps pooled patch by patch; each model's own map stays a member.

    A model's map is the equal-weight mean of its checkpoints' normalized attention and its
    probabilities follow its frozen ensemble rule. Interrupted model maps recompute whole.
    """
    models, pooling = plan["models"], plan.get("pooling")
    if (
        pooling not in POOLING_FUNCTIONS
        or len(models) < 2
        or any(
            not model["checkpoints"]
            or model["method"] not in {"refit", "ensemble", "seed_ensemble"}
            or (model["method"] == "refit" and len(model["checkpoints"]) != 1)
            for model in models
        )
        or plan["checkpoints"] != [c for model in models for c in model["checkpoints"]]
    ):
        raise ValueError(
            "Pooled attention needs two or more complete models and a pooling function."
        )
    if not plan["slides"] or len({row["slideId"] for row in plan["slides"]}) != len(plan["slides"]):
        raise ValueError("Select distinct nonempty slide identities.")
    verify_sources(plan["slides"])
    for checkpoint in plan["checkpoints"]:
        current = checkpoint_snapshot(checkpoint["path"], Path(checkpoint["path"]).parent)
        if any(current[key] != checkpoint[key] for key in ("path", "sha256", "bytes")):
            raise ValueError("A frozen predictor checkpoint changed.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(plan["resources"]["cpuThreadsPerRun"])
    torch.use_deterministic_algorithms(True)
    device = torch.device(plan.get("device", "cpu"))
    input_hash = content_hash(
        {
            key: plan.get(key)
            for key in (
                "method",
                "pooling",
                "models",
                "target",
                "featureContract",
                "slides",
                "code",
            )
        }
    )
    classes = len(plan["target"]["classes"])
    total_passes = len(plan["slides"]) * len(plan["checkpoints"])
    passes = 0

    def progress(slide_index, slide, model_index):
        write_json_atomic(
            folder / "progress.json",
            {
                "completedPairs": passes,
                "totalPairs": total_passes,
                "currentSlide": slide["slideId"],
                "completedSlides": slide_index,
                "totalSlides": len(plan["slides"]),
                "completedModels": model_index,
                "totalModels": len(models),
            },
        )

    artifacts, summaries = {}, []
    for slide_index, slide in enumerate(plan["slides"]):
        if (folder / "cancel.requested").exists():
            raise KeyboardInterrupt("Interpretation cancelled.")
        evidence, features, coords = inspect_inputs(
            slide, slide, plan["featureContract"], load=True
        )
        if any(slide.get(key) != value for key, value in evidence.items()):
            raise ValueError(
                "Feature vectors, coordinates, or geometry differ from reviewed evidence."
            )
        tensor = torch.from_numpy(features).unsqueeze(0).to(device)
        maps, model_probabilities, members = [], [], []
        for model_index, entry in enumerate(models):
            filename = f"slide-{slide_index}-member-{model_index}.json"
            path = folder / filename
            receipt = folder / f".slide-{slide_index}-member-{model_index}.receipt.json"
            weights = probabilities = None
            if receipt.exists() and path.exists():
                cached = json.loads(read_file_bounded(receipt, 4096))
                if cached.get("inputHash") == input_hash and cached.get("artifact") == _receipt(
                    path
                ):
                    value = json.loads(read_file_bounded(path, MAX_ARTIFACT_BYTES))
                    if (
                        value.get("slideId") == slide["slideId"]
                        and value.get("member") == str(model_index)
                        and len(value.get("patches", [])) == len(coords)
                    ):
                        weights = np.asarray(
                            [patch["weight"] for patch in value["patches"]], dtype=np.float64
                        )
                        probabilities = np.asarray(value["probabilities"], dtype=np.float64)
                        passes += len(entry["checkpoints"])
            if weights is None:
                mean_logit = entry.get("aggregation") == "mean_logit"
                count = len(entry["checkpoints"])
                weights = np.zeros(len(coords), dtype=np.float64)
                probabilities = np.zeros(classes, dtype=np.float64)
                log_probabilities = np.zeros(classes, dtype=np.float64)
                for checkpoint in entry["checkpoints"]:
                    if (folder / "cancel.requested").exists():
                        raise KeyboardInterrupt("Interpretation cancelled.")
                    w, p, log_p, note = _checkpoint_attention(
                        checkpoint, plan, tensor, slide, device, mean_logit
                    )
                    weights += w / count
                    probabilities += p / count
                    if mean_logit:
                        log_probabilities += log_p / count
                    passes += 1
                    progress(slide_index, slide, model_index)
                if mean_logit:
                    log_probabilities -= np.logaddexp.reduce(log_probabilities)
                    probabilities = np.exp(log_probabilities)
                weights = weights / weights.sum()
                _validate(weights, probabilities, len(coords), classes)
                # A member's map is that model's own, so its architecture describes it; the
                # pooled map below carries the pooling note.
                _write_attention_json(
                    path,
                    _map(
                        slide,
                        coords,
                        weights,
                        probabilities,
                        plan,
                        member=str(model_index),
                        note=note,
                    ),
                )
                write_json_atomic(receipt, {"inputHash": input_hash, "artifact": _receipt(path)})
            _validate(weights, probabilities, len(coords), classes)
            artifacts[filename] = _receipt(path)
            array_name = f"slide-{slide_index}-member-{model_index}.npy"
            _write_attention_array(folder / array_name, coords, weights)
            artifacts[array_name] = _receipt(folder / array_name)
            maps.append(weights)
            model_probabilities.append(probabilities)
            members.append(
                {
                    "index": model_index,
                    "predictorId": entry["predictorId"],
                    "name": entry.get("name"),
                    "method": entry["method"],
                    "checkpointCount": len(entry["checkpoints"]),
                    "probabilities": probabilities.tolist(),
                    "attentionArtifact": filename,
                    "attentionArray": array_name,
                }
            )
            progress(slide_index, slide, model_index + 1)
        del tensor, features
        pooled = POOLING_FUNCTIONS[pooling](np.stack(maps))
        pooled = pooled / pooled.sum()
        probabilities = np.mean(model_probabilities, axis=0)
        probabilities = probabilities / probabilities.sum()
        _validate(pooled, probabilities, len(coords), classes)
        filename = f"slide-{slide_index}.json"
        _write_attention_json(
            folder / filename,
            _map(
                slide,
                coords,
                pooled,
                probabilities,
                plan,
                member="mean",
                note=POOLED_ATTENTION_NOTE,
            ),
        )
        artifacts[filename] = _receipt(folder / filename)
        array_name = f"slide-{slide_index}.npy"
        _write_attention_array(folder / array_name, coords, pooled)
        artifacts[array_name] = _receipt(folder / array_name)
        summaries.append(
            {
                "slideId": slide["slideId"],
                "patchCount": len(coords),
                "probabilities": probabilities.tolist(),
                "attentionArtifact": filename,
                "attentionArray": array_name,
                "pooling": pooling,
                "members": members,
            }
        )
    verify_sources(plan["slides"])
    write_json_atomic(
        folder / "attention.json",
        {
            "runId": plan["runId"],
            "method": "pooled",
            "pooling": pooling,
            "classOrder": plan["target"]["classes"],
            "attentionKind": "class_independent_pooling",
            "attentionNote": POOLED_ATTENTION_NOTE,
            "slides": summaries,
        },
    )
    artifacts["attention.json"] = _receipt(folder / "attention.json")
    return {
        "state": "succeeded",
        "runId": plan["runId"],
        "method": "pooled",
        "pooling": pooling,
        "slideCount": len(summaries),
        "memberCount": len(models),
        "checkpointCount": len(plan["checkpoints"]),
        "classOrder": plan["target"]["classes"],
        "attentionKind": "class_independent_pooling",
        "attentionNote": POOLED_ATTENTION_NOTE,
        "slides": summaries,
        "artifacts": artifacts,
    }
