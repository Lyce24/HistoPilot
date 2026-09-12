"""Exact ABMIL whole-bag pooling attention for refit and frozen ensembles."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from histopilot.application.feature_bundles import _hash
from histopilot.application.predictors import checkpoint_snapshot
from histopilot.storage.attention_inputs import inspect_inputs, verify_sources
from histopilot.storage.packed import _source
from histopilot.storage.project_lock import _reject_symlink_components, ensure_managed_directory
from histopilot.storage.scientific import ScientificStore
from histopilot.training.module import MILTrainModule
from histopilot.workers.packing_process import write_json

ATTENTION_NOTE = (
    "Class-independent ABMIL pooling weights, normalized over all patches in this slide. "
    "The ensemble mean averages each member's normalized attention. Percentiles are "
    "within-slide midranks for display, not calibrated probabilities or class-specific attribution."
)
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024


def _write_attention_json(path, value):
    """Stream large maps atomically instead of using the 64-MiB metadata writer."""
    _reject_symlink_components(path)
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
    _reject_symlink_components(path)
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
    with _source(path) as (stream, stamp):
        if not 0 < stamp["sizeBytes"] <= MAX_ARTIFACT_BYTES:
            raise ValueError("Attention artifact is empty or exceeds its byte budget.")
        digest = hashlib.sha256()
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return {"path": str(path), "bytes": stamp["sizeBytes"], "sha256": digest.hexdigest()}


def _percentiles(weights):
    _, inverse, counts = np.unique(weights, return_inverse=True, return_counts=True)
    return ((np.cumsum(counts) - counts / 2) / len(weights))[inverse]


def _map(slide, coords, weights, probabilities, plan, *, member):
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
        "attentionNote": ATTENTION_NOTE,
        "classOrder": plan["target"]["classes"],
        "member": member,
        "probabilities": probabilities.tolist(),
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
        raise ValueError("ABMIL produced invalid attention or class probabilities.")


def interpret(plan, output_dir):
    """Resume completed slide/member pairs; interrupted pairs recompute the whole bag.

    Every attention value comes from the same complete-bag forward call as its
    member's class probabilities. There is no top-k filtering or local softmax.
    """
    folder = Path(output_dir)
    ensure_managed_directory(folder)
    method, checkpoints = plan["method"], plan["checkpoints"]
    if (
        method not in {"refit", "ensemble"}
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
    input_hash = _hash(
        {
            key: plan.get(key)
            for key in ("method", "target", "featureContract", "slides", "checkpoints", "code")
        }
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
        total_weights = np.zeros(len(coords), dtype=np.float64)
        total_probabilities = np.zeros(len(plan["target"]["classes"]), dtype=np.float64)
        members = []
        for member_index, checkpoint in enumerate(checkpoints):
            if (folder / "cancel.requested").exists():
                raise KeyboardInterrupt("Interpretation cancelled.")
            filename = f"slide-{slide_index}-member-{member_index}.json"
            path = folder / filename
            cache_receipt = folder / f".slide-{slide_index}-member-{member_index}.receipt.json"
            weights, probabilities = None, None
            if cache_receipt.exists() and path.exists():
                cached = json.loads(ScientificStore._read_file(cache_receipt, 4096))
                if cached.get("inputHash") != input_hash or cached.get("artifact") != _receipt(
                    path
                ):
                    raise ValueError(
                        "Saved attention member evidence changed; it cannot be resumed."
                    )
                value = json.loads(ScientificStore._read_file(path, MAX_ARTIFACT_BYTES))
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
            if weights is None:
                model = MILTrainModule.load_from_checkpoint(
                    checkpoint["path"], map_location="cpu", weights_only=True
                )
                if (
                    model.target != plan["target"]
                    or model.hparams.feature_dim != plan["featureContract"]["dimensions"]
                    or model.recipe.get("model", "abmil").lower() != "abmil"
                ):
                    raise ValueError(
                        "Checkpoint architecture, target or feature dimensions differ from the predictor."
                    )
                model.eval().to(device)
                with torch.inference_mode():
                    output = model.model(tensor, return_attention=True)
                    weights = output["attention"][0].double().cpu().numpy()
                    probabilities = (
                        torch.softmax(output["logits"][0].double(), dim=-1).cpu().numpy()
                    )
                del output, model
                # Double-precision renormalization removes softmax roundoff only.
                weights = weights / weights.sum()
                _validate(weights, probabilities, len(coords), len(total_probabilities))
                _write_attention_json(
                    path,
                    _map(slide, coords, weights, probabilities, plan, member=str(member_index)),
                )
                write_json(cache_receipt, {"inputHash": input_hash, "artifact": _receipt(path)})
            _validate(weights, probabilities, len(coords), len(total_probabilities))
            artifacts[filename] = _receipt(path)
            array_name = f"slide-{slide_index}-member-{member_index}.npy"
            _write_attention_array(folder / array_name, coords, weights)
            artifacts[array_name] = _receipt(folder / array_name)
            total_weights += weights / len(checkpoints)
            total_probabilities += probabilities / len(checkpoints)
            members.append(
                {
                    "index": member_index,
                    "checkpointSha256": checkpoint["sha256"],
                    "probabilities": probabilities.tolist(),
                    "attentionArtifact": filename,
                    "attentionArray": array_name,
                }
            )
            write_json(
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
        _validate(total_weights, total_probabilities, len(coords), len(total_probabilities))
        filename = f"slide-{slide_index}.json"
        _write_attention_json(
            folder / filename,
            _map(slide, coords, total_weights, total_probabilities, plan, member="mean"),
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
    write_json(
        folder / "attention.json",
        {
            "runId": plan["runId"],
            "method": method,
            "classOrder": plan["target"]["classes"],
            "attentionKind": "class_independent_pooling",
            "attentionNote": ATTENTION_NOTE,
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
        "attentionNote": ATTENTION_NOTE,
        "slides": summaries,
        "artifacts": artifacts,
    }
