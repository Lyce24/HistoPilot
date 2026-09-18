"""Frozen clinical covariates and patient-level, fitting-partition preprocessing.

No spreadsheet is reopened here. Values come from the immutable dataset snapshot
and are carried in a checksummed execution plan. Every ensemble member retains
its own training-only transform in its checkpoint.
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


def clinical_rows(memberships, values, fields):
    """Validate the declared schema and consistency of patient-level covariates."""
    patients = {}
    result = []
    for row in memberships:
        slide, patient = row["slideId"], row.get("patientId")
        if not patient or row.get("patientIdSource") == "slide_fallback":
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
        if patient in patients and patients[patient] != converted:
            raise ValueError(f"Patient {patient} has conflicting clinical values across slides.")
        patients[patient] = converted
        result.append(converted)
    return result


def fit_clinical_preprocessor(memberships, values, fields):
    """Fit medians/scales/category vocabulary from training patients exactly once."""
    if not fields or len({item["field"] for item in fields}) != len(fields):
        raise ValueError("Clinical modeling requires distinct explicitly typed fields.")
    if not memberships or any(row.get("partition") != "train" for row in memberships):
        raise ValueError("Clinical preprocessing may fit only the training partition.")
    rows = clinical_rows(memberships, values, fields)
    patients = {row["patientId"]: value for row, value in zip(memberships, rows, strict=True)}
    ordered = [patients[key] for key in sorted(patients)]
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
                    "observedPatients": len(observed),
                }
            )
            names.extend([f"{name}:standardized", f"{name}:missing"])
        else:
            categories = sorted(set(observed))
            if len(categories) > 256:
                raise ValueError(f"Clinical field {name} has more than 256 training categories.")
            columns.append({**item, "categories": categories, "observedPatients": len(observed)})
            names.extend([f"{name}:{category}" for category in categories])
            names.extend([f"{name}:missing", f"{name}:unknown"])
    return {
        "version": 1,
        "method": "training_patient_median_standardize_onehot_v1",
        "columns": columns,
        "featureNames": names,
        "dimensions": len(names),
        "trainingPatientCount": len(patients),
        "trainingPatientIdsSha256": hashlib.sha256(
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
