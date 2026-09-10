"""Explicit scientific source pools, before any cross-validation assignments."""

from collections import Counter

from histopilot.application.modern_splits import _json, _subset, modern_assignments

ALGORITHM_V3 = "histopilot-explicit-pools-evaluation-v3"
POOL_ROLES = ("train", "val", "test")


def select_pools(groups, pools, evaluator, finding, fixed_assignments):
    direct = {role: [] for role in POOL_ROLES}
    expanded = {role: [] for role in POOL_ROLES}
    if pools.source == "rules":
        assignments, direct, expanded = fixed_assignments(groups, pools.rules, evaluator, finding)
        if not pools.rules.test:
            finding(
                "TEST_POOL_REQUIRED",
                "Define the external test pool before previewing the strategy.",
            )
        if pools.trainSelection == "rules" and not pools.rules.train:
            finding(
                "TRAIN_POOL_REQUIRED",
                "Define the training pool, or explicitly choose all remaining groups.",
            )
        if pools.validationSource == "fixed" and not pools.rules.val:
            finding(
                "VALIDATION_POOL_REQUIRED",
                "Define the fixed validation pool, or choose a percentage of training.",
            )
        if pools.trainSelection == "remaining":
            for patient, rows in sorted(groups.items()):
                if patient not in assignments:
                    assignments[patient] = "train"
                    expanded["train"].extend(rows)
    else:
        assignments = {}
        imported = pools.imported
        for patient, rows in sorted(groups.items()):
            raw_values = {evaluator.field(row, imported.partitionField) for row in rows}
            if any(value is None or value not in imported.partitionLabels for value in raw_values):
                finding(
                    "INVALID_IMPORTED_PARTITION",
                    "Map a pool for every included slide in the predefined partition column.",
                )
                continue
            roles = {imported.partitionLabels[value] for value in raw_values}
            roles = {"train" if role == "trainval" else role for role in roles}
            if len(roles) != 1:
                finding(
                    "IMPORTED_PATIENT_LEAKAGE",
                    "Predefined pool values put slides from one group in different pools.",
                )
                continue
            role = next(iter(roles))
            assignments[patient] = role
            direct[role].extend(rows)
            expanded[role].extend(rows)
    if pools.validationSource == "training_fraction" and expanded["val"]:
        finding(
            "FIXED_VALIDATION_NOT_SELECTED",
            "Predefined validation groups exist. Choose fixed validation or map those groups into training.",
        )
    for role in POOL_ROLES:
        required = role in {"train", "test"} or pools.validationSource == "fixed"
        if required and not expanded[role]:
            finding(
                f"{role.upper()}_POOL_EMPTY",
                f"The selected {role} pool contains no eligible groups.",
            )
    remaining = [
        row
        for patient, rows in sorted(groups.items())
        if patient not in assignments
        for row in rows
    ]
    if remaining:
        finding(
            "UNASSIGNED_POOL_GROUPS",
            "Some eligible groups belong to no pool. Assign them or explicitly choose remaining groups for training.",
        )
    return assignments, direct, expanded, remaining


def pool_counts(groups, assignments, classes):
    result = {}
    for role in POOL_ROLES:
        patients = [patient for patient in sorted(groups) if assignments.get(patient) == role]
        rows = [row for patient in patients for row in groups[patient]]
        counts = Counter(groups[patient][0]["label"] for patient in patients)
        result[role] = {
            "patients": sum(
                groups[patient][0].get("patientIdSource") != "slide_fallback"
                for patient in patients
            ),
            "groups": len(patients),
            "slides": len(rows),
            "fallbackSlides": sum(row.get("patientIdSource") == "slide_fallback" for row in rows),
            "classes": {label: counts[label] for label in classes},
        }
    return result


def explicit_assignments(spec, groups, assignments, evaluator, finding, max_memberships, max_bytes):
    split = spec.split
    training = {
        patient: rows for patient, rows in groups.items() if assignments.get(patient) == "train"
    }
    validation = {
        patient: rows for patient, rows in groups.items() if assignments.get(patient) == "val"
    }
    external_test = {patient for patient in groups if assignments.get(patient) == "test"}
    fixed_validation = validation if split.pools.validationSource == "fixed" else None
    final_rows = sum(map(len, groups.values())) * len(split.seeds)
    # Every final plan contains all included rows. Reserve its bounded space before
    # expanding CV, including long identities and the v3 pool membership marker.
    final_bytes = sum(
        len(
            _json(
                {
                    "seed": 4294967295,
                    "fold": None,
                    "planId": "seed:4294967295/final",
                    "phase": "final",
                    "pool": "external_test",
                    "partition": "train",
                    "slideId": row["slideId"],
                    "patientId": patient,
                    "patientIdSource": row.get("patientIdSource", "source"),
                    "label": row["label"],
                }
            )
        )
        + 1
        for patient, rows in groups.items()
        for row in rows
    ) * len(split.seeds)
    if final_rows > max_memberships:
        finding(
            "PROTOCOL_MEMBERSHIP_LIMIT",
            "The final evaluation plans exceed 500,000 explicit membership rows.",
        )
        return [], training
    if final_bytes > max_bytes:
        finding(
            "PROTOCOL_DOCUMENT_LIMIT",
            "The final evaluation memberships exceed the 15 MiB protocol limit.",
        )
        return [], training
    plans = []
    if split.mode != "held_out":
        blocked_bounds = []

        def bounded_finding(code, message, severity="error"):
            if code in {"PROTOCOL_DOCUMENT_LIMIT", "PROTOCOL_MEMBERSHIP_LIMIT"}:
                blocked_bounds.append(code)
            finding(code, message, severity)

        plans = modern_assignments(
            spec,
            training,
            {},
            evaluator,
            bounded_finding,
            max_memberships - final_rows,
            max_bytes - final_bytes,
            fixed_validation=fixed_validation,
        )
        if blocked_bounds:
            return [], training
        plans = [({**metadata, "pool": "training"}, members) for metadata, members in plans]
    for seed in split.seeds:
        val = (
            set(validation)
            if fixed_validation is not None
            else _subset(
                set(training),
                split.validationFraction,
                seed,
                ["final", "early_stop"],
                spec,
                training,
                finding,
                role="early-stop validation",
            )
        )
        members = {
            **{patient: "train" for patient in training if patient not in val},
            **{patient: "val" for patient in val},
            **{patient: "test" for patient in external_test},
        }
        if set(members) != set(groups):
            finding(
                "INCOMPLETE_ASSIGNMENTS",
                "Every included group must belong to one role in final evaluation.",
            )
        plans.append(
            (
                {
                    "seed": seed,
                    "fold": None,
                    "planId": f"seed:{seed}/final",
                    "phase": "final",
                    "pool": "external_test",
                },
                members,
            )
        )
    return plans, training
