"""Frozen clinical covariates and fitting-partition preprocessing per design unit.

No spreadsheet is reopened here. Values come from the immutable dataset snapshot
and are carried in a checksummed execution plan. Every ensemble member retains
its own training-only transform in its checkpoint.

Patient-level designs treat covariates as patient attributes: every slide of a
patient must agree, identities must be verified, and each training patient counts
once. Slide-level designs treat each slide as its own unit, so covariates may vary
between a patient's slides and no patient identity is needed.
"""

import hashlib
import json
import math
import statistics


def clinical_fields(recipe):
    return recipe.get("clinicalFields", []) if recipe.get("inputMode", "image") != "image" else []


def _canonical(value, kind):
    if value is None or isinstance(value, str) and not value.strip():
        return None
    if kind == "numeric":
        if isinstance(value, bool):
            raise ValueError("Boolean values need a categorical clinical field.")
        try:
            number = float(value)
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError(
                "Numeric clinical fields require finite numbers or missing values."
            ) from error
        if not math.isfinite(number) or abs(number) > 1e30:
            raise ValueError("Numeric clinical values must be finite and within float32 range.")
        return number
    if kind != "categorical" or type(value) not in (str, int, float, bool):
        raise ValueError("Categorical clinical fields require scalar values.")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Categorical clinical values must be finite.")
    token = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(token) > 4096:
        raise ValueError("Categorical clinical values cannot exceed 4096 characters.")
    return token


def clinical_rows(memberships, values, fields, *, unit="patient"):
    """Validate the declared schema, and patient consistency for patient-level designs."""
    patients = {}
    result = []
    for row in memberships:
        slide, patient = row["slideId"], row.get("patientId")
        if unit != "slide" and (not patient or row.get("patientIdSource") == "slide_fallback"):
            raise ValueError("Clinical modeling requires verified patient identities.")
        if slide not in values:
            raise ValueError(f"Clinical values are absent for slide {slide}.")
        raw = values[slide]
        if not isinstance(raw, dict) or any(item["field"] not in raw for item in fields):
            raise ValueError(
                f"Slide {slide} lacks a declared clinical field; map the field explicitly."
            )
        converted = {}
        for item in fields:
            try:
                converted[item["field"]] = _canonical(raw[item["field"]], item["kind"])
            except ValueError as error:
                raise ValueError(f"{slide}, {item['field']}: {error}") from error
        if unit != "slide":
            if patient in patients and patients[patient] != converted:
                raise ValueError(
                    f"Patient {patient} has conflicting clinical values across slides."
                )
            patients[patient] = converted
        result.append(converted)
    return result


def fit_clinical_preprocessor(memberships, values, fields, *, unit="patient"):
    """Fit medians/scales/category vocabulary from the training units, each once."""
    if not fields or len({item["field"] for item in fields}) != len(fields):
        raise ValueError("Clinical modeling requires distinct explicitly typed fields.")
    if not memberships or any(row.get("partition") != "train" for row in memberships):
        raise ValueError("Clinical preprocessing may fit only the training partition.")
    rows = clinical_rows(memberships, values, fields, unit=unit)
    key = "slideId" if unit == "slide" else "patientId"
    patients = {row[key]: value for row, value in zip(memberships, rows, strict=True)}
    ordered = [patients[key] for key in sorted(patients)]
    # Patient designs keep their original receipt keys, which saved checkpoints compare.
    noun = "Slide" if unit == "slide" else "Patient"
    columns, names = [], []
    for item in fields:
        name, kind = item["field"], item["kind"]
        observed = [row[name] for row in ordered if row[name] is not None]
        if kind == "numeric":
            median = float(statistics.median(observed)) if observed else 0.0
            filled = [median if row[name] is None else row[name] for row in ordered]
            mean = math.fsum(filled) / len(filled)
            scale = (
                math.sqrt(math.fsum((value - mean) ** 2 for value in filled) / len(filled)) or 1.0
            )
            columns.append(
                {
                    **item,
                    "median": median,
                    "mean": mean,
                    "scale": scale,
                    f"observed{noun}s": len(observed),
                }
            )
            names.extend([f"{name}:standardized", f"{name}:missing"])
        else:
            categories = sorted(set(observed))
            if len(categories) > 256:
                raise ValueError(f"Clinical field {name} has more than 256 training categories.")
            columns.append({**item, "categories": categories, f"observed{noun}s": len(observed)})
            names.extend([f"{name}:{category}" for category in categories])
            names.extend([f"{name}:missing", f"{name}:unknown"])
    return {
        "version": 1,
        "method": f"training_{noun.lower()}_median_standardize_onehot_v1",
        "columns": columns,
        "featureNames": names,
        "dimensions": len(names),
        f"training{noun}Count": len(patients),
        f"training{noun}IdsSha256": hashlib.sha256(
            json.dumps(sorted(patients)).encode()
        ).hexdigest(),
        "trainingValuesSha256": hashlib.sha256(
            json.dumps(patients, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
    }


def transform_clinical(rows, preprocessor):
    """Apply an already fitted transform, with distinct missing/unseen categories."""
    if preprocessor.get("version") != 1 or not preprocessor.get("columns"):
        raise ValueError("A saved clinical preprocessing contract is required.")
    result = []
    for row in rows:
        features = []
        for column in preprocessor["columns"]:
            field = column["field"]
            if field not in row:
                raise ValueError(
                    f"Required clinical field {field} is absent; explicit null is missing."
                )
            value = _canonical(row[field], column["kind"])
            if column["kind"] == "numeric":
                scale = column["scale"]
                if not math.isfinite(scale) or scale <= 0:
                    raise ValueError("Saved clinical scales must be positive and finite.")
                filled = column["median"] if value is None else value
                features.extend([(filled - column["mean"]) / scale, float(value is None)])
            else:
                categories = column["categories"]
                features.extend(float(value == category) for category in categories)
                features.extend(
                    [float(value is None), float(value is not None and value not in categories)]
                )
        if len(features) != preprocessor["dimensions"] or any(
            not math.isfinite(value) or abs(value) > 1e30 for value in features
        ):
            raise ValueError(
                "Clinical transformation produced invalid feature dimensions or values."
            )
        result.append(features)
    return result


def label_separation(memberships, values, item, positive=None, *, unit="patient"):
    """How well one covariate alone separates the labels, or None when it cannot be judged.

    Categorical fields report the share of units whose category's majority label is
    their own, counting only categories with at least three units. Numeric fields report
    the better of AUROC and 1 - AUROC for binary targets. A value near 1 suggests the
    field restates the label, for example a reader grade behind a consensus target.
    """
    key = "slideId" if unit == "slide" else "patientId"
    units = {}
    for row in memberships:
        try:
            value = _canonical((values.get(row["slideId"]) or {}).get(item["field"]), item["kind"])
        except ValueError:
            return None
        if value is not None and row.get("label") is not None:
            units.setdefault(row.get(key) or row["slideId"], (value, row["label"]))
    if len(units) < 10 or len({label for _, label in units.values()}) < 2:
        return None
    if item["kind"] == "categorical":
        groups = {}
        for value, label in units.values():
            groups.setdefault(value, []).append(label)
        groups = {value: labels for value, labels in groups.items() if len(labels) >= 3}
        covered = sum(len(labels) for labels in groups.values())
        if len(groups) < 2 or covered < len(units) / 2:
            return None
        majority = sum(max(labels.count(label) for label in set(labels)) for labels in groups.values())
        return majority / covered
    if positive is None:
        return None
    ranked = sorted(units.values(), key=lambda pair: pair[0])
    positives = sum(label == positive for _, label in ranked)
    negatives = len(ranked) - positives
    if not positives or not negatives:
        return None
    rank_sum, index = 0.0, 0
    while index < len(ranked):
        tied = index
        while tied + 1 < len(ranked) and ranked[tied + 1][0] == ranked[index][0]:
            tied += 1
        average_rank = (index + tied) / 2 + 1
        rank_sum += average_rank * sum(label == positive for _, label in ranked[index : tied + 1])
        index = tied + 1
    auroc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    return max(auroc, 1 - auroc)
