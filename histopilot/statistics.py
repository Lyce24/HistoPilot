"""Patient bootstrap inference, shared by compute workers and the lightweight API.

Ordinary (not class-stratified) percentile bootstrap resamples independent
patients. Paired differences reuse each draw for both models. Intervals condition
on the fitted models and observed prediction procedure, not model retraining.
"""

import hashlib
import json
import math

import numpy as np

from histopilot.schemas.analysis import PatientAnalysisSettings
from histopilot.scoring import class_ranking_score

METHOD = "patient_percentile_bootstrap_v1"
LIMITATION = (
    "95% percentile intervals resample patients with replacement. They condition on the "
    "fitted models and do not include training, model selection, or threshold uncertainty. "
    "Replicates missing a required class are excluded and counted."
)


def _settings(value):
    return PatientAnalysisSettings.model_validate(value or {}).model_dump()


def _patients(records, target):
    """Validate and canonically align already-aggregated patient predictions."""
    classes = target["classes"]
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("Patient analysis requires distinct frozen classes.")
    identities = [row.get("patientId") for row in records]
    if any(not isinstance(value, str) or not value.strip() for value in identities):
        raise ValueError("Patient analysis requires verified patient identities.")
    if len(set(identities)) != len(identities):
        raise ValueError("Patient analysis requires exactly one prediction per patient.")
    if any(row.get("patientIdSource") == "slide_fallback" for row in records):
        raise ValueError("Slide-ID fallback cannot establish independent patients.")
    rows = sorted(records, key=lambda row: row["patientId"])
    for row in rows:
        label = row.get("labelIndex")
        if label is not None and (
            type(label) is not int
            or not 0 <= label < len(classes)
            or row.get("label") != classes[label]
        ):
            raise ValueError("Patient labels must preserve the frozen class order.")
        probability = np.asarray(row.get("probabilities"), dtype=float)
        if (
            probability.shape != (len(classes),)
            or not np.isfinite(probability).all()
            or (probability < 0).any()
            or (probability > 1).any()
            or not np.isclose(probability.sum(), 1, atol=1e-5, rtol=0)
        ):
            raise ValueError("Patient probabilities must be finite and normalized.")
        if "logProbabilities" in row:
            logs = np.asarray(row["logProbabilities"], dtype=float)
            if (
                logs.shape != probability.shape
                or not np.isfinite(logs).all()
                or not np.isclose(np.logaddexp.reduce(logs), 0, atol=1e-5, rtol=0)
                or not np.allclose(np.exp(logs), probability, atol=1e-6, rtol=1e-5)
            ):
                raise ValueError("Patient log probabilities must match saved probabilities.")
    return rows


class _Ranking:
    """Sort once; bootstrap multiplicities preserve ties without sorting each draw."""

    def __init__(self, rows, target):
        labels = np.asarray([row["labelIndex"] for row in rows], dtype=int)
        indices = (
            [target["classes"].index(target["positiveClass"])]
            if target["task"] == "binary_classification"
            else range(len(target["classes"]))
        )
        self.groups = []
        for index in indices:
            scores = np.asarray([class_ranking_score(row, index) for row in rows])
            if np.isnan(scores).any():
                raise ValueError("Patient ranking scores cannot be NaN.")
            order = np.argsort(-scores, kind="stable")
            ordered = scores[order]
            starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
            self.groups.append((order, labels[order] == index, starts))

    def __call__(self, weights):
        values = []
        for order, positive, starts in self.groups:
            ordered_weights = weights[order]
            tp = np.add.reduceat(ordered_weights * positive, starts)
            fp = np.add.reduceat(ordered_weights * ~positive, starts)
            p, n = tp.sum(), fp.sum()
            if not p or not n:
                return None
            ctp, cfp = np.cumsum(tp), np.cumsum(fp)
            auc = np.sum(tp * (n - cfp + 0.5 * fp)) / (p * n)
            precision = np.divide(ctp, ctp + cfp, out=np.zeros_like(ctp), where=ctp + cfp > 0)
            ap = np.sum(tp * precision) / p
            values.append((auc, ap))
        return np.mean(values, axis=0)


def patient_bootstrap(records, target, settings=None, *, other=None):
    """Return point estimates, 95% CIs, and optionally paired left-minus-right CIs."""
    policy = _settings(settings)
    left = _patients(records, target)
    right = _patients(other, target) if other is not None else None
    if right is not None and [(row["patientId"], row.get("labelIndex")) for row in left] != [
        (row["patientId"], row.get("labelIndex")) for row in right
    ]:
        raise ValueError("Paired comparisons require exactly the same patients and labels.")
    labeled = [index for index, row in enumerate(left) if row.get("labelIndex") is not None]
    rows = [left[index] for index in labeled]
    count = len(rows)
    counts = {
        name: sum(row["labelIndex"] == i for row in rows)
        for i, name in enumerate(target["classes"])
    }
    base = {
        "method": METHOD,
        "confidenceLevel": policy["confidenceLevel"],
        "seed": policy["bootstrapSeed"],
        "resamples": policy["bootstrapResamples"],
        "patientCount": count,
        "unlabeledCount": len(left) - count,
        "classCounts": counts,
        "paired": right is not None,
        "scope": "fixed_predictions",
        "note": LIMITATION,
        "validResamples": 0,
        "excludedResamples": 0,
    }
    if not count or not all(counts.values()):
        return {**base, "available": False, "reason": "Every frozen class needs labeled patients."}
    rankers = [_Ranking(rows, target)]
    if right is not None:
        rankers.append(_Ranking([right[index] for index in labeled], target))
    estimates = [ranker(np.ones(count)) for ranker in rankers]
    point = dict(zip(("auroc", "auprc"), map(float, estimates[0]), strict=True))
    if right is not None:
        point = {
            name: {
                "left": float(estimates[0][i]),
                "right": float(estimates[1][i]),
                "difference": float(estimates[0][i] - estimates[1][i]),
            }
            for i, name in enumerate(("auroc", "auprc"))
        }
    base["estimates"] = point
    if min(counts.values()) < 2:
        return {
            **base,
            "available": False,
            "reason": "Bootstrap intervals require at least two patients in every class.",
        }
    rng = np.random.default_rng(policy["bootstrapSeed"])
    draws = []
    for _ in range(policy["bootstrapResamples"]):
        weights = np.bincount(rng.integers(count, size=count), minlength=count).astype(float)
        values = [ranker(weights) for ranker in rankers]
        if any(value is None for value in values):
            continue
        draws.append(values[0] if right is None else values[0] - values[1])
    base.update(validResamples=len(draws), excludedResamples=base["resamples"] - len(draws))
    if len(draws) < max(100, math.ceil(base["resamples"] * 0.8)):
        return {
            **base,
            "available": False,
            "reason": "Too few bootstrap replicates contain all required classes.",
        }
    alpha = (1 - policy["confidenceLevel"]) / 2
    limits = np.quantile(draws, [alpha, 1 - alpha], axis=0, method="linear")
    return {
        **base,
        "available": True,
        "intervals": {
            name: {"lower": float(limits[0, i]), "upper": float(limits[1, i])}
            for i, name in enumerate(("auroc", "auprc"))
        },
        "degenerate": bool(np.any(np.all(np.asarray(draws) == draws[0], axis=0))),
    }


def one_slide_per_patient(slides, seed):
    """Choose by a seeded identity hash, before consulting outcomes or scores."""
    groups, seen = {}, set()
    for row in slides:
        patient, slide = row.get("patientId"), row.get("slideId")
        if not patient or row.get("patientIdSource") == "slide_fallback":
            raise ValueError("One-slide sensitivity analysis requires verified patient IDs.")
        if not slide or slide in seen:
            raise ValueError("One-slide sensitivity analysis requires unique slide IDs.")
        seen.add(slide)
        key = hashlib.sha256(
            json.dumps(
                ["one-slide-per-patient-v1", seed, patient, slide], separators=(",", ":")
            ).encode()
        ).hexdigest()
        if patient not in groups or key < groups[patient][0]:
            groups[patient] = (key, row)
    return [groups[patient][1] for patient in sorted(groups)]


def patient_analysis(slides, patients, target, settings=None):
    policy = _settings(settings)
    try:
        selected = one_slide_per_patient(slides, policy["oneSlideSeed"])
        if {row["patientId"] for row in selected} != {row["patientId"] for row in patients}:
            raise ValueError("Patient predictions must cover every selected slide's patient.")
        primary = patient_bootstrap(patients, target, policy)
        # A patient's observed outcome is shared even if the selected slide has
        # no individual label. Selection itself never consults outcome values.
        labels = {row["patientId"]: row for row in patients}
        sensitivity_rows = [
            {
                **row,
                "label": labels[row["patientId"]].get("label"),
                "labelIndex": labels[row["patientId"]].get("labelIndex"),
            }
            for row in selected
        ]
        sensitivity = patient_bootstrap(sensitivity_rows, target, policy)
        return {
            "policy": policy,
            "uncertainty": primary,
            "oneSlidePerPatient": {
                "method": "seeded_identity_sha256_v1",
                "seed": policy["oneSlideSeed"],
                "slideIds": [row["slideId"] for row in selected],
                "patientCount": len(selected),
                "metrics": sensitivity.get("estimates"),
                "uncertainty": sensitivity,
            },
        }
    except ValueError as error:
        unavailable = {"available": False, "reason": str(error)}
        return {
            "policy": policy,
            "uncertainty": unavailable,
            "oneSlidePerPatient": {"uncertainty": unavailable},
        }
