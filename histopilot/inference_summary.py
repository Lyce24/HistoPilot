"""Label-free descriptions of saved predictions.

Nothing here reads outcomes. The inference worker freezes a summary with these
functions, and the control service recomputes views from checksummed
predictions without importing the training runtime.
"""

import math
from collections import Counter

BINS = 20
THRESHOLD_SWEEP = tuple(round(0.05 * step, 2) for step in range(1, 20))
NEAR_THRESHOLD_BANDS = (0.05, 0.1)


def binary_positive(target):
    classes = target["classes"]
    if len(classes) == 2 and target.get("positiveClass") in classes:
        return classes.index(target["positiveClass"])
    return None


def predicted_index(probabilities, target, threshold):
    """The frozen decision rule: binary thresholds, otherwise the first highest class."""
    positive = binary_positive(target)
    if positive is not None:
        return positive if probabilities[positive] >= threshold else 1 - positive
    return max(range(len(probabilities)), key=lambda index: probabilities[index])


def decision_margin(probabilities, target, threshold):
    """Distance from the frozen decision boundary on a 0-1 scale.

    Binary targets measure how far the positive probability lies from the frozen
    threshold, relative to the room on that side of it; other targets use the top
    class minus the runner-up. Both equal |2p - 1| for a binary threshold of 0.5.
    """
    positive = binary_positive(target)
    if positive is not None:
        score = probabilities[positive]
        if score >= threshold:
            return (score - threshold) / (1 - threshold)
        return (threshold - score) / threshold
    ordered = sorted(probabilities, reverse=True)
    return ordered[0] - ordered[1] if len(ordered) > 1 else 1.0


def member_agreement(row, target, threshold, index):
    members = row.get("memberProbabilities")
    if not isinstance(members, list) or len(members) < 2:
        return None
    votes = [predicted_index(values, target, threshold) for values in members]
    chosen = [values[index] for values in members]
    mean = math.fsum(chosen) / len(chosen)
    spread = math.sqrt(math.fsum((value - mean) ** 2 for value in chosen) / len(chosen))
    return {"agree": sum(vote == index for vote in votes), "total": len(votes), "spread": spread}


def describe(row, target, threshold):
    """Per-record decision, confidence, margin and member agreement."""
    probabilities = row["probabilities"]
    index = predicted_index(probabilities, target, threshold)
    result = {
        "predictedIndex": index,
        "predictedLabel": target["classes"][index],
        "confidence": probabilities[index],
        "margin": decision_margin(probabilities, target, threshold),
    }
    agreement = member_agreement(row, target, threshold, index)
    if agreement is not None:
        result["memberAgreement"] = agreement
    return result


def quantile(ordered, fraction):
    """Linear interpolation between order statistics (numpy's default rule)."""
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def histogram_bin(value, bins=BINS):
    return min(bins - 1, max(0, int(value * bins)))


def distribution(values, groups, labels, bins=BINS):
    """Quantiles plus a unit-interval histogram stacked by predicted class."""
    ordered = sorted(values)
    counts = {label: [0] * bins for label in labels}
    for value, group in zip(values, groups, strict=True):
        counts[group][histogram_bin(value, bins)] += 1
    return {
        "mean": math.fsum(values) / len(values) if values else None,
        "quantiles": {
            name: quantile(ordered, fraction)
            for name, fraction in (
                ("p10", 0.1), ("p25", 0.25), ("median", 0.5), ("p75", 0.75), ("p90", 0.9),
            )
        },
        "edges": [round(step / bins, 4) for step in range(bins + 1)],
        "counts": counts,
    }


def summarize(rows, target, threshold, *, described=None):
    """A frozen, label-free summary of one prediction unit."""
    classes = target["classes"]
    described = described or [describe(row, target, threshold) for row in rows]
    total = len(described)
    labels = [item["predictedLabel"] for item in described]
    counts = Counter(labels)
    confidence = [item["confidence"] for item in described]
    result = {
        "count": total,
        "predicted": [
            {
                "label": label,
                "count": counts[label],
                "fraction": counts[label] / total if total else None,
                "meanConfidence": (
                    math.fsum(item["confidence"] for item in described if item["predictedLabel"] == label)
                    / counts[label]
                    if counts[label]
                    else None
                ),
            }
            for label in classes
        ],
        "confidence": distribution(confidence, labels, classes),
        "margin": distribution([item["margin"] for item in described], labels, classes),
    }
    positive = binary_positive(target)
    if positive is not None:
        scores = [row["probabilities"][positive] for row in rows]
        histogram = [0] * BINS
        for score in scores:
            histogram[histogram_bin(score)] += 1
        sweep = sorted({*THRESHOLD_SWEEP, threshold})
        result["binary"] = {
            "positiveClass": classes[positive],
            "threshold": threshold,
            "positiveProbability": {
                "edges": [round(step / BINS, 4) for step in range(BINS + 1)],
                "counts": histogram,
            },
            "sweep": [
                {"threshold": value, "positive": sum(score >= value for score in scores)}
                for value in sweep
            ],
            "nearThreshold": [
                {"band": band, "count": sum(abs(score - threshold) < band for score in scores)}
                for band in NEAR_THRESHOLD_BANDS
            ],
        }
    agreements = [item["memberAgreement"] for item in described if "memberAgreement" in item]
    if agreements:
        members = max(item["total"] for item in agreements)
        votes = Counter(item["agree"] for item in agreements)
        result["ensemble"] = {
            "memberCount": members,
            "records": len(agreements),
            "agreement": [
                {"agree": agree, "count": votes[agree]} for agree in range(members, -1, -1)
            ],
            "unanimous": sum(item["agree"] == item["total"] for item in agreements),
            "disagreements": sum(item["agree"] < item["total"] for item in agreements),
            "meanSpread": math.fsum(item["spread"] for item in agreements) / len(agreements),
        }
    return result


def class_counts(described, classes):
    counts = Counter(item["predictedLabel"] for item in described)
    return {label: counts[label] for label in classes}


def cross_tab(described, keys, classes, *, limit=50):
    """Predicted class counts for each group value, largest groups first."""
    groups = {}
    for item, key in zip(described, keys, strict=True):
        groups.setdefault(key, []).append(item)
    ordered = sorted(groups.items(), key=lambda entry: (-len(entry[1]), str(entry[0])))
    rows = [
        {
            "value": key,
            "count": len(items),
            "counts": class_counts(items, classes),
            "meanConfidence": math.fsum(item["confidence"] for item in items) / len(items),
        }
        for key, items in ordered[:limit]
    ]
    remainder = [item for _, items in ordered[limit:] for item in items]
    return {
        "rows": rows,
        "otherValues": max(0, len(ordered) - limit),
        **({"other": {"count": len(remainder), "counts": class_counts(remainder, classes)}}
           if remainder else {}),
    }


def agreement(left, right, classes):
    """Label-free agreement between two decision lists over the same records."""
    size = len(classes)
    matrix = [[0] * size for _ in range(size)]
    for a, b in zip(left, right, strict=True):
        matrix[a][b] += 1
    total = len(left)
    same = sum(matrix[index][index] for index in range(size))
    observed = same / total if total else None
    expected = (
        math.fsum(
            (sum(matrix[index]) / total) * (sum(row[index] for row in matrix) / total)
            for index in range(size)
        )
        if total
        else None
    )
    kappa = (
        (observed - expected) / (1 - expected)
        if total and expected is not None and expected < 1
        else None
    )
    return {
        "count": total,
        "agreement": observed,
        "kappa": kappa,
        "disagreements": total - same,
        "matrix": matrix,
    }


def patient_member_probabilities(slides, aggregation):
    """Each member's patient probabilities under the frozen slide-combination rule."""
    members = [row.get("memberProbabilities") for row in slides]
    if (
        not members
        or any(not isinstance(value, list) or len(value) < 2 for value in members)
        or len({len(value) for value in members}) != 1
    ):
        return None
    logs = [row.get("memberLogProbabilities") for row in slides]
    has_logs = all(value is not None for value in logs)
    if aggregation == "mean_logits" and not has_logs and any(
        value == 0 for slide in members for member in slide for value in member
    ):
        # Legacy probability-only evidence cannot recover an underflowed logit.
        # Omit agreement rather than inventing patient votes by clipping it.
        return None
    result = []
    for member in range(len(members[0])):
        vectors = [value[member] for value in members]
        if aggregation == "mean_logits":
            log_vectors = (
                [value[member] for value in logs] if has_logs
                else [[math.log(value) for value in vector] for vector in vectors]
            )
            average = [math.fsum(column) / len(vectors) for column in zip(*log_vectors, strict=True)]
            peak = max(average)
            shift = peak + math.log(math.fsum(math.exp(value - peak) for value in average))
            result.append([math.exp(value - shift) for value in average])
        else:
            result.append(
                [math.fsum(column) / len(vectors) for column in zip(*vectors, strict=True)]
            )
    return result
