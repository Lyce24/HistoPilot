"""Configuration selection from validation receipts, independent of assessment outcomes."""

import math


def validation_selection(plan, state):
    metric = plan.get("selectionMetric")
    if metric is None:
        return None
    key = metric.removeprefix("validation_")
    unit = plan["target"]["unit"]
    states = {row["id"]: row for row in state["runs"]}
    candidates = []
    for candidate in plan["configurations"]:
        runs = [row for row in plan["runs"] if row["candidateId"] == candidate["id"]]
        scores = []
        for run in runs:
            actual = states.get(run["id"], {})
            detail = ((actual.get("result") or {}).get("metrics") or {}).get("validation") or {}
            score = detail.get(unit, {}).get(key)
            if (
                actual.get("status") == "completed"
                and detail.get("unit") == unit
                and detail.get(unit, {}).get("available") is not False
                and detail.get("patientAggregation")
                == candidate["recipe"].get("patientAggregation", "mean_probabilities")
                and isinstance(score, (int, float))
                and not isinstance(score, bool)
                and math.isfinite(score)
                and score >= 0
                and (key == "loss" or score <= 1)
            ):
                scores.append({"runId": run["id"], "value": score})
        complete = bool(runs) and len(scores) == len(runs)
        candidates.append(
            {
                "candidateId": candidate["id"],
                "number": candidate["number"],
                "complete": complete,
                "completedScores": len(scores),
                "expectedScores": len(runs),
                "score": math.fsum(row["value"] for row in scores) / len(scores)
                if complete
                else None,
                "foldScores": scores,
            }
        )
    ready = bool(candidates) and all(row["complete"] for row in candidates)
    direction = "min" if key == "loss" else "max"
    winner = (
        min(
            candidates,
            key=lambda row: (
                row["score"] * (1 if direction == "min" else -1),
                row["number"],
                row["candidateId"],
            ),
        )
        if ready
        else None
    )
    return {
        "metric": metric,
        "unit": unit,
        "direction": direction,
        "source": "mean_fold_validation_across_training_and_split_seeds",
        "tieBreak": "configuration_number_then_id",
        "ready": ready,
        "selectedCandidateId": winner["candidateId"] if winner else None,
        "candidates": candidates,
        "note": "Configuration selection uses validation predictions at each selected checkpoint. Assessment and external outcomes are excluded. OOF intervals condition on fitted models; they do not account for configuration selection.",
    }
