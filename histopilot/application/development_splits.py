"""Development-only source selection and versioned plans.

The existing `test` assignment key describes an assessment fold internally. It
never identifies an external inference cohort in this version. Legacy algorithms
and frozen documents are left unchanged.
"""

from histopilot.application.modern_splits import modern_assignments

ALGORITHM_V4 = "histopilot-development-plans-v4"
ROLES = ("train", "val", "test")


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
        {},
        evaluator,
        finding,
        max_rows,
        max_bytes,
        fixed_validation=validation if spec.split.pools.validationSource == "fixed" else None,
    )
    return [({**metadata, "pool": "development"}, members) for metadata, members in plans], training


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
