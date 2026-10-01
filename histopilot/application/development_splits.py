"""Development-only source selection and versioned plans.

The existing `test` assignment key describes an assessment fold internally. It
never identifies an external inference cohort in this version. Frozen protocols
of earlier split versions keep their stored memberships; nothing regenerates them.
"""

from histopilot.application.modern_splits import group_class_counts, modern_assignments

ALGORITHM_V4 = "histopilot-development-plans-v4"
ALGORITHM_V4_MIXED = "histopilot-development-labelset-plans-v4"
ROLES = ("train", "val", "test")
# Designs that train: every unit is assessed at most once per split seed, so the
# out-of-fold predictions of a seed pool into one result.
TRAINABLE_MODES = ("kfold", "predefined_folds", "leave_one_domain_out", "held_out")


def training_split_issue(split):
    """``(code, message)`` when a development training design cannot train, else None.

    Monte Carlo repeats assess a unit in several plans of one seed, and nested designs
    need a search inside each outer fold; both stay available for planning only. A
    held-out assessment trains with one split seed, so its seeds share one assessment set.
    """
    mode = split.get("mode")
    if mode == "nested_kfold":
        return (
            "NESTED_SELECTION_REQUIRED",
            "Nested CV requires a separate search and selected refit inside each outer fold. "
            "Batch planning for that dependency is not connected yet.",
        )
    if mode not in TRAINABLE_MODES:
        return (
            "TRAINING_SPLIT_UNSUPPORTED",
            "Monte Carlo designs assess a unit in several plans of one split seed, so their "
            "out-of-fold predictions cannot be pooled. Choose k-fold, predefined folds, "
            "leave-one-site-out or a held-out assessment.",
        )
    if mode == "held_out" and len(split.get("seeds") or ()) != 1:
        return (
            "TRAINING_SPLIT_UNSUPPORTED",
            "A held-out assessment trains with one split seed. Repeat it over training seeds, "
            "or choose k-fold to assess every unit.",
        )
    return None


def select_development_pools(groups, pools, evaluator, finding, fixed_assignments):
    direct = {role: [] for role in ROLES}
    expanded = {role: [] for role in ROLES}
    assignments = {}
    if pools.rules.test or (pools.imported and "test" in pools.imported.partitionLabels.values()):
        finding("DEVELOPMENT_SCOPE_REQUIRED", "Select development and validation groups only.")
        return assignments, direct, expanded, [row for rows in groups.values() for row in rows]
    if pools.source == "rules":
        assignments, direct, expanded = fixed_assignments(groups, pools.rules, evaluator, finding)
        if pools.trainSelection == "rules" and not pools.rules.train:
            finding("TRAIN_POOL_REQUIRED", "Define training conditions or use all eligible groups.")
        if pools.validationSource == "fixed" and not pools.rules.val:
            finding("VALIDATION_POOL_REQUIRED", "Define fixed validation conditions.")
        if pools.trainSelection == "remaining":
            for patient, rows in sorted(groups.items()):
                if patient not in assignments:
                    assignments[patient] = "train"
                    expanded["train"].extend(rows)
    else:
        imported = pools.imported
        for patient, rows in sorted(groups.items()):
            selected = [
                (row, imported.partitionLabels.get(evaluator.field(row, imported.partitionField)))
                for row in rows
            ]
            roles = {"train" if role == "trainval" else role for _, role in selected if role}
            if len(roles) > 1:
                finding(
                    "IMPORTED_PATIENT_LEAKAGE",
                    "Predefined values put slides from one group in different development pools.",
                )
                continue
            if roles:
                role = next(iter(roles))
                assignments[patient] = role
                direct[role].extend(row for row, selected_role in selected if selected_role)
                expanded[role].extend(rows)
    if pools.validationSource == "training_fraction" and expanded["val"]:
        finding(
            "FIXED_VALIDATION_NOT_SELECTED",
            "Choose fixed validation for mapped validation groups, or map them into training.",
        )
    for role in ("train", "val"):
        if (role == "train" or pools.validationSource == "fixed") and not expanded[role]:
            finding(f"{role.upper()}_POOL_EMPTY", f"The selected {role} pool contains no groups.")
    # Unmatched source rows are outside this development protocol, not a reserved
    # evaluation dataset. Their labels and feature coverage are not prerequisites.
    remaining = [
        row
        for patient, rows in sorted(groups.items())
        if patient not in assignments
        for row in rows
    ]
    return assignments, direct, expanded, remaining


def development_assignments(spec, groups, assignments, evaluator, finding, max_rows, max_bytes):
    training = {
        patient: rows for patient, rows in groups.items() if assignments.get(patient) == "train"
    }
    validation = {
        patient: rows for patient, rows in groups.items() if assignments.get(patient) == "val"
    }
    plans = modern_assignments(
        spec,
        training,
        evaluator,
        finding,
        max_rows,
        max_bytes,
        fixed_validation=validation if spec.split.pools.validationSource == "fixed" else None,
    )
    return [({**metadata, "pool": "development"}, members) for metadata, members in plans], training


def pool_counts(groups, assignments, classes, *, split_unit="patient"):
    result = {}
    for role in ROLES:
        patients = [patient for patient in sorted(groups) if assignments.get(patient) == role]
        rows = [row for patient in patients for row in groups[patient]]
        counts = group_class_counts(groups, set(patients))
        result[role] = {
            "patients": 0
            if split_unit == "slide"
            else sum(
                groups[patient][0].get("patientIdSource") != "slide_fallback"
                for patient in patients
            ),
            "groups": len(patients),
            "slides": len(rows),
            "fallbackSlides": sum(row.get("patientIdSource") == "slide_fallback" for row in rows),
            "classes": {label: counts[label] for label in classes},
        }
    return result


def development_summary(spec):
    return {
        "scope": "development",
        "finalPlanCount": 0,
        "validationSource": spec.split.pools.validationSource,
        "evaluationScope": "Development assessment for comparing and selecting configurations",
        "validationFractionScope": "Fraction of fitting groups after development assessment and inner tuning groups are removed; fixed validation overrides it",
        "roleDescriptions": {
            "train": "Model fitting within the development cohort",
            "val": "Development validation for early stopping",
            "test": "Development assessment fold; excluded from fitting and early stopping",
            "tune": "Inner development fold for configuration selection; outer assessment groups are excluded",
        },
    }
