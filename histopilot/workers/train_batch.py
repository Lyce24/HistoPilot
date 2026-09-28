"""Fold planning, the single-fold compute entry point and OOF results collection.

The Task Center runs each fold with ``histopilot.workers.managed_fold`` and collects the
batch with ``histopilot.workers.managed_collect``; both import their work from here.
"""

from __future__ import annotations

import hashlib
import json
import traceback
from collections import Counter, defaultdict
from pathlib import Path

from histopilot.application.feature_bundles import _hash
from histopilot.workers.packing_process import write_json
from histopilot.workers.training_process import (
    classify_training_failure,
    compute_snapshot,
    now,
    read_json,
)


def _run_plan(batch: dict, run: dict, gpu: int | None) -> dict:
    from histopilot.schemas.nnmil import resolve_nnmil_plan
    from histopilot.schemas.training_controls import sampling_memberships

    candidate = next(item for item in batch["configurations"] if item["id"] == run["candidateId"])
    split = next(item for item in batch["splitPlans"] if item["id"] == run["splitPlanId"])
    plan = {
        "runId": run["id"],
        "batchId": batch["batchId"],
        "batchContentHash": batch["batchContentHash"],
        "code": batch.get("code"),
        "runtime": batch["runtime"],
        "candidateId": run["candidateId"],
        "trainingSeed": run["trainingSeed"],
        "splitPlan": split,
        "recipe": candidate["recipe"],
        "target": batch["target"],
        **({"splitUnit": batch["splitUnit"]} if "splitUnit" in batch else {}),
        "resources": batch["resources"],
        "device": "cpu" if gpu is None else "cuda",
        "data": {
            **batch["data"],
            "memberships": sampling_memberships(
                batch["memberships"][run["splitPlanId"]],
                candidate["recipe"],
                batch["data"].get("cohortValues", {}),
            ),
        },
    }
    return resolve_nnmil_plan(plan)


def _check_inputs(data):
    from histopilot.storage.pack_import import pack_file_stamps
    from histopilot.storage.packed import _check_sources

    _check_sources({"sourceStamps": data["sourceStamps"]})
    if data.get("packPath") and pack_file_stamps(Path(data["packPath"])) != data["packStamps"]:
        raise ValueError("The verified feature pack changed after the batch was launched.")


def _prediction_records(path, target):
    value = json.loads(Path(path).read_text())
    if isinstance(value, dict) and value.get("classOrder") != target["classes"]:
        raise ValueError("Prediction class order differs from the frozen target.")
    return value if isinstance(value, list) else value["records"]


def collect_results(batch: dict, state: dict, folder: Path):
    from histopilot.candidate_selection import validation_selection

    selection = validation_selection(batch, state)
    groups = defaultdict(list)
    splits = {row["id"]: row for row in batch["splitPlans"]}
    state_runs = {row["id"]: row for row in state["runs"]}
    if len(state_runs) != len(state["runs"]):
        raise ValueError("A training run occurs more than once in the batch state.")
    runs = state["runs"]
    if "runs" in batch:
        planned_runs = {row["id"]: row for row in batch["runs"]}
        if len(planned_runs) != len(batch["runs"]) or set(state_runs) - set(planned_runs):
            raise ValueError("Batch state contains runs outside the frozen execution plan.")
        runs = []
        for identity, planned in planned_runs.items():
            actual = state_runs.get(identity)
            if actual is not None and any(
                actual.get(key) != planned.get(key)
                for key in ("candidateId", "trainingSeed", "splitPlanId")
            ):
                raise ValueError("A training run's candidate, seed or fold changed.")
            runs.append(actual if actual is not None else {**planned, "status": "pending"})
    for run in runs:
        groups[
            (run["candidateId"], run["trainingSeed"], splits[run["splitPlanId"]]["seed"])
        ].append(run)
    candidates, oof = [], []
    for (candidate, training_seed, split_seed), runs in groups.items():
        completed = [run for run in runs if run["status"] == "completed"]
        item = {
            "candidateId": candidate,
            "trainingSeed": training_seed,
            "splitSeed": split_seed,
            "completedRuns": len(completed),
            "totalRuns": len(runs),
            "complete": len(completed) == len(runs),
            "metrics": None,
            "oofPath": None,
        }
        if item["complete"]:
            from histopilot.training.module import classification_metrics

            records, expected, patient_folds = [], {}, {}
            for run in runs:
                records.extend(
                    _prediction_records(run["result"]["predictions"]["assessment"], batch["target"])
                )
                for row in batch["memberships"][run["splitPlanId"]]:
                    if row["partition"] == "test":
                        if row["slideId"] in expected:
                            raise ValueError(
                                "An assessment slide appears in multiple folds of one k-fold seed."
                            )
                        expected[row["slideId"]] = row
                        # Patient-grouped folds, including slide targets, keep a patient in one fold.
                        if batch.get("splitUnit") != "slide" or batch.get("groupByPatient"):
                            patient = row.get("patientId")
                            previous = patient_folds.setdefault(patient, run["splitPlanId"])
                            if previous != run["splitPlanId"]:
                                raise ValueError("An assessment patient appears in multiple folds of one k-fold seed.")
            actual = Counter(row["slideId"] for row in records)
            if set(actual) != set(expected) or any(count != 1 for count in actual.values()):
                raise ValueError(
                    "OOF predictions must cover each assessment-eligible slide exactly once."
                )
            for record in records:
                expected_row = expected[record["slideId"]]
                if (
                    record["patientId"] != expected_row["patientId"]
                    or record["label"] != expected_row["label"]
                    or type(record.get("labelIndex")) is not int
                    or record["labelIndex"]
                    != batch["target"]["classes"].index(expected_row["label"])
                ):
                    raise ValueError(
                        "An OOF prediction identity or label differs from frozen memberships."
                    )
                source = expected_row.get("patientIdSource")
                if "patientIdSource" in record and record["patientIdSource"] != source:
                    raise ValueError("An OOF patient identity source differs from frozen membership.")
                if "patientIdSource" in expected_row:
                    record["patientIdSource"] = source
            recipe = next(
                row["recipe"] for row in batch["configurations"] if row["id"] == candidate
            )
            key = hashlib.sha256(f"{candidate}/{training_seed}/{split_seed}".encode()).hexdigest()[
                :24
            ]
            path = folder / f"oof-{key}.json"
            # Completed groups are collected repeatedly while other groups train.
            # Reuse analysis only when the actual predictions and scoring policy match.
            analysis_hash = _hash({"records": records, "target": batch["target"],
                                   "recipe": recipe, "code": batch.get("code"),
                                   **({"splitUnit": batch["splitUnit"]} if "splitUnit" in batch else {})})
            cached = None
            if path.exists():
                try:
                    cached = read_json(path)
                except (OSError, ValueError):
                    pass
            summary = cached.get("summary") if (
                isinstance(cached, dict) and cached.get("analysisInputHash") == analysis_hash
            ) else None
            if isinstance(summary, dict):
                try:
                    if cached.get("analysisSummaryHash") != _hash(summary):
                        summary = None
                except (TypeError, ValueError):
                    summary = None
            if not isinstance(summary, dict):
                summary = classification_metrics(
                    records, batch["target"], recipe.get("patientAggregation", "mean_probabilities"),
                    analysis=recipe.get("analysis"),
                    decision_threshold=recipe.get("decisionThreshold", 0.5),
                    split_unit=batch.get("splitUnit"),
                )
            write_json(
                path,
                {
                    "batchId": batch["batchId"],
                    "candidateId": candidate,
                    "trainingSeed": training_seed,
                    "splitSeed": split_seed,
                    "protocolId": batch["protocolId"],
                    "classOrder": batch["target"]["classes"],
                    "records": records,
                    "summary": summary,
                    "analysisInputHash": analysis_hash,
                    "analysisSummaryHash": _hash(summary),
                    "purpose": "development_assessment",
                },
            )
            item.update(
                metrics=summary["selected"],
                metricDetails=summary,
                oofPath=str(path),
                assessmentSlideCount=len(expected),
            )
            oof.append(
                {
                    "path": str(path),
                    "candidateId": candidate,
                    "trainingSeed": training_seed,
                    "splitSeed": split_seed,
                    "slideCount": len(expected),
                }
            )
        candidates.append(item)
        if selection:
            candidate_score = next(row for row in selection["candidates"]
                                   if row["candidateId"] == candidate)
            item.update(selectionScore=candidate_score["score"],
                        selected=selection["selectedCandidateId"] == candidate)
    write_json(
        folder / "results.json",
        {
            "batchId": batch["batchId"],
            "status": state["status"],
            "candidates": candidates,
            "oof": oof,
            **({"selection": selection} if selection else {}),
            "selectionNote": "OOF metrics describe development assessment. Comparing hyperparameters on these folds does not provide an independent final-test estimate.",
        },
    )


def run_fold_worker(plan_path: Path):
    plan = read_json(plan_path)
    folder = plan_path.parent
    try:
        if plan.get("code") is not None and plan["code"] != compute_snapshot():
            raise ValueError(
                "Training code changed after this execution was prepared. Clone a new batch."
            )
        _check_inputs(plan["data"])
        from histopilot.training.fold import train_fold

        checkpoint = folder / "last.ckpt"
        result = train_fold(
            plan, folder, checkpoint_path=checkpoint if checkpoint.exists() else None
        )
        _check_inputs(plan["data"])
        write_json(folder / "result.json", result)
    except BaseException as error:
        write_json(
            folder / "failure.json",
            {
                "error": str(error),
                "type": type(error).__name__,
                "category": classify_training_failure(str(error)),
                "traceback": traceback.format_exc(),
                "at": now(),
            },
        )
        raise
