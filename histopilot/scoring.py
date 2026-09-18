"""Stable classification ranking without importing the training runtime."""

import math
from collections import defaultdict

import numpy as np


def patient_predictions(records, aggregation="mean"):
    """Aggregate validated slide predictions without importing the compute runtime.

    Both external inference and control-service exports use this same rule.
    Missing patient IDs are omitted; explicit slide-ID fallback is rejected so
    it cannot be represented as independent patient evidence.
    """
    if aggregation not in {"mean", "mean_logits"}:
        raise ValueError("Choose mean probabilities or mean logits for patient aggregation.")
    groups = defaultdict(list)
    for row in records:
        if row.get("patientIdSource") == "slide_fallback":
            raise ValueError("Patient scoring requires verified patient IDs, not slide-ID fallback.")
        if row["patientId"]:
            groups[row["patientId"]].append(row)
    patients = []
    for patient, slides in sorted(groups.items()):
        labels = {row["labelIndex"] for row in slides if row["labelIndex"] is not None}
        if len(labels) > 1:
            raise ValueError("Patient evaluation requires consistent labels within each patient.")
        labeled = next((row for row in slides if row["labelIndex"] is not None), None)
        patients.append({
            "patientId": patient,
            "slideIds": [row["slideId"] for row in slides],
            "labelIndex": labeled["labelIndex"] if labeled else None,
            "label": labeled["label"] if labeled else None,
            "probabilities": np.mean([row["probabilities"] for row in slides], axis=0).tolist(),
        })
        if all("logProbabilities" in row for row in slides):
            patients[-1]["logProbabilities"] = (
                np.logaddexp.reduce([row["logProbabilities"] for row in slides], axis=0)
                - np.log(len(slides))
            ).tolist()
        if aggregation == "mean_logits":
            if not all("logProbabilities" in row for row in slides):
                raise ValueError("Mean-logit patient scoring requires saved log probabilities.")
            average = np.mean([row["logProbabilities"] for row in slides], axis=0)
            normalized = average - np.logaddexp.reduce(average)
            patients[-1].update(
                logProbabilities=normalized.tolist(), probabilities=np.exp(normalized).tolist()
            )
    return patients


def logsumexp(values):
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))


def class_ranking_score(row, index):
    """One-versus-rest log odds from validated prediction evidence.

    Log odds preserve probability ordering even when probabilities round to
    zero or one. Using only log(p) still loses ordering near one; the saved
    log probabilities of the other classes retain that information. This works
    after either probability or logit averaging, including multiclass scores.
    Legacy records have only their rounded probabilities available.
    """
    logs = row.get("logProbabilities")
    if logs is not None:
        return logs[index] - logsumexp(
            [value for other, value in enumerate(logs) if other != index]
        )
    probabilities = row["probabilities"]
    rest = math.fsum(value for other, value in enumerate(probabilities) if other != index)
    # Exact legacy endpoints are tied extrema. Do not floor a small *nonzero*
    # probability: doing so would erase ordering already present in the file.
    # Scores are used only for ranking; reports serialize probability cutoffs.
    if probabilities[index] == 0:
        return -math.inf
    if rest == 0:
        return math.inf
    return math.log(probabilities[index]) - math.log(rest)
