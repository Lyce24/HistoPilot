"""Authenticated, Torch-free exports of complete frozen OOF prediction groups."""

import csv
import hashlib
import io
import json

from histopilot.application.clinical import _validate_records
from histopilot.application.development import _plan_metadata
from histopilot.application.feature_bundles import _hash
from histopilot.application.predictors import read_evidence
from histopilot.schemas.nnmil import resolve_nnmil_plan
from histopilot.schemas.training_controls import sampling_memberships
from histopilot.scoring import patient_predictions
from histopilot.storage.project_lock import StorageError


def _invalid(message):
    return StorageError(message, "TRAINING_OOF_INVALID", 409)


def _identity(value):
    if not isinstance(value, str) or not value or value in {".", ".."} or any(
        char in value for char in ("/", "\\", "\0")
    ):
        raise _invalid("A saved training identity is invalid.")
    return value


def training_oof_csv(store, batch_id, candidate_id, training_seed, split_seed, unit):
    """Export a completed candidate/seed group using its frozen scoring policy."""
    batch = store.get_configuration(batch_id)
    manifest = batch["manifest"]
    if manifest.get("kind") != "mil-batch":
        raise StorageError("Choose a frozen development batch.", "INVALID_BATCH", 422)
    if unit not in {"slide", "patient"}:
        raise StorageError("Choose patient or slide predictions.", "TRAINING_OOF_UNIT_INVALID", 422)
    candidate = next((row for row in manifest["configurations"] if row["id"] == candidate_id), None)
    splits = {row["id"]: row for row in manifest["splitPlans"] if row["seed"] == split_seed}
    runs = [row for row in manifest["runs"] if row["candidateId"] == candidate_id
            and row["trainingSeed"] == training_seed and row["splitPlanId"] in splits]
    if candidate is None or not runs:
        raise StorageError("This configuration and seed group is not in the batch.",
                           "TRAINING_OOF_NOT_FOUND", 404)
    if len(runs) != len(splits) or {row["splitPlanId"] for row in runs} != set(splits):
        raise _invalid("The frozen candidate must contain every fold of the selected seed.")
    folder = store.folder / "training" / _identity(batch_id)
    try:
        plan = read_evidence(folder / "plan.json", folder)
        state = read_evidence(folder / "state.json", folder)
        protocol_id = manifest["spec"]["inputs"]["protocolId"]
        protocol = store.get_configuration(protocol_id)
        target = protocol["manifest"]["spec"]["target"]
        classes, recipe = target["classes"], candidate["recipe"]
        if (plan.get("batchId") != batch_id or plan.get("batchContentHash") != batch["contentHash"]
                or plan.get("protocolId") != protocol_id
                or plan.get("protocolContentHash") != protocol["contentHash"]
                or plan.get("target") != target
                or plan.get("configurations") != manifest["configurations"]
                or plan.get("runs") != manifest["runs"]
                or plan.get("splitPlans") != manifest["splitPlans"]):
            raise _invalid("The saved execution plan differs from the frozen batch.")
        states = {row["id"]: row for row in state["runs"]}
        if len(states) != len(state["runs"]):
            raise _invalid("Duplicate run states cannot establish complete OOF evidence.")
        expected, fold_records, patient_folds = {}, {}, {}
        for run in runs:
            current = states.get(run["id"], {})
            if current.get("status") != "completed":
                raise StorageError("Every fold in this seed group must finish before export.",
                                   "TRAINING_OOF_INCOMPLETE", 409)
            run_folder = folder / "runs" / _identity(run["id"])
            receipt = read_evidence(run_folder / "result.json", run_folder)
            run_plan = read_evidence(run_folder / "plan.json", run_folder)
            members = [row for row in protocol["manifest"]["memberships"]
                       if _hash(_plan_metadata(row)) == run["splitPlanId"]]
            expected_data = {**plan["data"], "memberships": sampling_memberships(
                members, recipe, plan["data"].get("cohortValues", {})
            )}
            resolved = resolve_nnmil_plan({"recipe": recipe, "data": expected_data})
            if (receipt != current.get("result") or receipt.get("state") != "succeeded"
                    or receipt.get("runId") != run["id"]
                    or run_plan.get("runId") != run["id"]
                    or run_plan.get("batchId") != batch_id
                    or run_plan.get("batchContentHash") != batch["contentHash"]
                    or run_plan.get("candidateId") != candidate_id
                    or run_plan.get("code") != plan.get("code")
                    or run_plan.get("runtime") != plan.get("runtime")
                    or run_plan.get("recipe") != recipe or run_plan.get("target") != target
                    or run_plan.get("data") != expected_data
                    or any(run_plan.get(key) != resolved.get(key)
                           or receipt.get(key) != resolved.get(key)
                           for key in ("effectiveRecipe", "nnmilPlanning"))
                    or run_plan.get("trainingSeed") != training_seed
                    or run_plan.get("splitPlan") != splits[run["splitPlanId"]]
                    or plan["memberships"].get(run["splitPlanId"]) != members):
                raise _invalid("A completed fold differs from its frozen execution evidence.")
            evidence = read_evidence(receipt["predictions"]["assessment"], run_folder)
            if (not isinstance(receipt.get("bestCheckpointPath"), str)
                    or evidence.get("checkpointPath") != receipt["bestCheckpointPath"]):
                raise _invalid("Fold predictions do not identify their selected checkpoint.")
            if evidence.get("classOrder") != classes:
                raise _invalid("Fold prediction class order differs from the frozen target.")
            records = _validate_records(evidence["records"], classes)
            assessment = {row["slideId"]: row for row in members if row["partition"] == "test"}
            if {row["slideId"] for row in records} != set(assessment):
                raise _invalid("Fold predictions must cover exactly its assessment slides.")
            for row in records:
                identity = row["slideId"]
                member = assessment[identity]
                if identity in expected or (
                    row.get("patientId") != member["patientId"] or row.get("label") != member["label"]
                ):
                    raise _invalid("OOF identities and labels must match each assessment fold exactly once.")
                patient = member["patientId"]
                fold = splits[run["splitPlanId"]]["fold"]
                if patient in patient_folds and patient_folds[patient] != fold:
                    raise _invalid("One patient's slides appear in different assessment folds.")
                patient_folds[patient] = fold
                expected[identity], fold_records[identity] = member, row
        key = hashlib.sha256(f"{candidate_id}/{training_seed}/{split_seed}".encode()).hexdigest()[:24]
        document = read_evidence(folder / f"oof-{key}.json", folder)
        if any(document.get(field) != value for field, value in {
            "batchId": batch_id, "candidateId": candidate_id, "trainingSeed": training_seed,
            "splitSeed": split_seed, "protocolId": protocol_id, "classOrder": classes,
            "purpose": "development_assessment",
        }.items()):
            raise _invalid("The OOF artifact belongs to a different frozen experiment group.")
        records = _validate_records(document["records"], classes)
        if set(row["slideId"] for row in records) != set(expected):
            raise _invalid("OOF predictions are incomplete.")
        expected_hash = _hash({"records": records, "target": target, "recipe": recipe,
                               "code": plan.get("code")})
        if document.get("analysisInputHash") != expected_hash:
            raise _invalid("OOF predictions or their frozen scoring policy changed.")
        slides = []
        for row in records:
            member = expected[row["slideId"]]
            source = member.get("patientIdSource")
            for value in (row, fold_records[row["slideId"]]):
                if "patientIdSource" in value and value["patientIdSource"] != source:
                    raise _invalid("Patient identity provenance differs from the frozen protocol.")
            current = {**row, "patientIdSource": source}
            original = {**fold_records[row["slideId"]], "patientIdSource": source}
            if current != original:
                raise _invalid("OOF predictions differ from their completed fold artifacts.")
            slides.append(current)
        aggregation = "mean_logits" if recipe.get("patientAggregation") == "mean_logits" else "mean"
        if unit == "patient" and any(not row.get("patientId") for row in slides):
            raise _invalid("Patient exports require a verified patient identity for every slide.")
        selected = patient_predictions(slides, aggregation) if unit == "patient" else slides
        selected.sort(key=lambda row: row["patientId" if unit == "patient" else "slideId"])
        fields = (["patientId", "slideIds"] if unit == "patient" else ["slideId", "patientId"])
        fields += ["label", "predictedLabel", "assessmentFold", "trainingSeed", "splitSeed"]
        fields += [f"probability:{label}" for label in classes]
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        threshold = recipe.get("decisionThreshold", 0.5)
        for row in selected:
            probabilities = row["probabilities"]
            predicted = max(range(len(classes)), key=lambda index: probabilities[index])
            if target["task"] == "binary_classification" and threshold is not None:
                positive = classes.index(target["positiveClass"])
                predicted = positive if probabilities[positive] >= threshold else 1 - positive
            value = {name: row.get(name) for name in fields[:2]}
            if unit == "patient":
                value["slideIds"] = json.dumps(sorted(row["slideIds"]), ensure_ascii=False)
            value.update(label=row["label"], predictedLabel=classes[predicted],
                         assessmentFold=patient_folds[row["patientId"]],
                         trainingSeed=training_seed, splitSeed=split_seed)
            value.update({f"probability:{label}": probability
                          for label, probability in zip(classes, probabilities, strict=True)})
            writer.writerow(value)
        return stream.getvalue().encode("utf-8")
    except StorageError as error:
        if error.code.startswith("TRAINING_OOF"):
            raise
        raise _invalid(str(error)) from error
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        raise _invalid("Saved OOF evidence is missing, invalid, or inconsistent.") from error
