"""Versioned patient-grouped evaluation plans, distinct from early stopping.

All random orders derive from the frozen algorithm, seed and plan context. Inner
plans never contain an outer test group. No fitting or model choice occurs here.
"""

import hashlib
import json
import math
from collections import Counter, defaultdict

ALGORITHM_V2 = "histopilot-patient-evaluation-v2"
ROLES = ("train", "val", "test", "tune")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _ordered(patients, seed, context):
    return sorted(
        patients,
        key=lambda patient: (
            hashlib.sha256(_json([ALGORITHM_V2, seed, context, patient])).digest(),
            patient,
        ),
    )


def _strata(patients, groups, stratify):
    result = defaultdict(list)
    for patient in sorted(patients):
        result[groups[patient][0]["label"] if stratify else "all"].append(patient)
    return result


def _subset(patients, fraction, seed, context, spec, groups, finding, *, role):
    """Nearest per-stratum fraction, preserving feasible minimum group counts."""
    selected = set()
    for label, members in sorted(_strata(patients, groups, spec.split.stratify).items()):
        ordered = _ordered(members, seed, [context, label])
        requested_count = math.floor(len(ordered) * fraction + 0.5)
        count = requested_count
        minimum = (
            spec.constraints.minPatientsPerClass
            if spec.split.stratify
            else spec.constraints.minPatientsPerPartition
        )
        if len(ordered) >= 2 * minimum:
            count = min(len(ordered) - minimum, max(minimum, count))
        if count != requested_count:
            stratum = f" in class '{label}'" if spec.split.stratify else ""
            finding(
                "SPLIT_FRACTION_ADJUSTED",
                f"Seed {seed}, {context[0]}: {role} requests {fraction * 100:.6g}% "
                f"({requested_count} rounded groups) from {len(ordered)} groups{stratum}; "
                f"minimum group constraints allocate {count} groups.",
                "warning",
            )
        selected.update(ordered[:count])
    return selected


def _folds(patients, count, seed, context, spec, groups, finding):
    if len(patients) < count:
        finding("INFEASIBLE_FOLDS", "There are fewer groups than requested folds.")
        return None
    result = [set() for _ in range(count)]
    for label, members in sorted(_strata(patients, groups, spec.split.stratify).items()):
        ordered = _ordered(members, seed, [context, label])
        offset = min(range(count), key=lambda fold: (len(result[fold]), fold))
        for index, patient in enumerate(ordered):
            result[(offset + index) % count].add(patient)
    return result


def _domain_groups(spec, groups, evaluator, finding):
    domains = {}
    for patient, rows in sorted(groups.items()):
        values = {evaluator.field(row, spec.split.domainField) for row in rows}
        if any(value is None or not str(value).strip() for value in values):
            finding("MISSING_DOMAIN", "Every included group needs a site or cohort value.")
        elif len(values) != 1:
            finding(
                "INCONSISTENT_PATIENT_DOMAIN",
                "All slides from a group must belong to the same site or cohort.",
            )
        else:
            domains[patient] = next(iter(values))
    observed = sorted(set(domains.values()))
    if len(observed) < 2:
        finding("INSUFFICIENT_DOMAINS", "Leave-one-site/cohort-out requires at least two domains.")
    if len(observed) > 100:
        finding("DOMAIN_LIMIT", "This preview supports at most 100 sites or cohorts.")
    selected = (
        sorted(spec.split.heldOutDomains) if spec.split.domainPolicy == "selected" else observed
    )
    if set(selected) - set(observed):
        finding(
            "UNKNOWN_HELD_OUT_DOMAIN",
            "A selected held-out domain is absent from the training pool."
            if spec.split.version == 3
            else "A selected held-out domain is absent from the eligible cohort.",
        )
    return domains, selected


def _imported_groups(spec, groups, evaluator, finding):
    imported = spec.split.imported
    result = {}
    for patient, rows in sorted(groups.items()):
        values = {evaluator.field(row, imported.partitionField) for row in rows}
        if any(value is None or value not in imported.partitionLabels for value in values):
            finding(
                "INVALID_IMPORTED_PARTITION",
                "Every included slide needs an explicitly mapped partition.",
            )
            continue
        roles = {imported.partitionLabels[value] for value in values}
        # trainval is a development designation; val is kept only when explicit.
        roles = {"train" if role == "trainval" else role for role in roles}
        if len(roles) != 1:
            finding(
                "IMPORTED_PATIENT_LEAKAGE",
                "An imported split puts slides from one group in different partitions.",
            )
        else:
            result[patient] = next(iter(roles))
    return result


def modern_assignments(
    spec, groups, fixed, evaluator, finding, max_memberships, max_bytes, *, fixed_validation=None
):
    """Return bounded plan descriptors and assignments, then shared summary metadata."""
    split = spec.split
    patients = set(groups)
    validation_groups = fixed_validation or {}
    all_groups = {**groups, **validation_groups}
    domains, held_out_domains = {}, []
    if split.mode == "leave_one_domain_out":
        domains, held_out_domains = _domain_groups(spec, groups, evaluator, finding)
        for rows in validation_groups.values():
            values = {evaluator.field(row, split.domainField) for row in rows}
            if any(value is None or not str(value).strip() for value in values):
                finding(
                    "MISSING_DOMAIN", "Every fixed validation group needs a site or cohort value."
                )
            elif len(values) != 1:
                finding(
                    "INCONSISTENT_PATIENT_DOMAIN",
                    "All slides of a fixed validation group must belong to one site or cohort.",
                )
        per_seed = len(held_out_domains)
    elif split.mode == "nested_kfold":
        per_seed = split.outerFolds * (split.innerFolds + 1)
    elif split.mode == "kfold":
        per_seed = split.folds
    elif split.mode == "monte_carlo":
        per_seed = split.repeats
    else:
        per_seed = 1
    plan_count = per_seed * len(split.seeds)
    row_count = sum(map(len, all_groups.values()))
    if row_count * plan_count > max_memberships:
        finding(
            "PROTOCOL_MEMBERSHIP_LIMIT",
            "The requested plans exceed 500,000 explicit membership rows. Reduce folds, repeats, seeds or cohort size.",
        )
        return []
    # Include metadata and identifiers before allocating any plan/membership arrays.
    row_bytes = sum(
        len(
            _json(
                {
                    "seed": 4294967295,
                    "fold": 9,
                    "planId": "seed:4294967295/outer:9/inner:9",
                    "phase": "evaluation",
                    "outerFold": 9,
                    "innerFold": 9,
                    "partition": "train",
                    "slideId": row["slideId"],
                    "patientId": patient,
                    "patientIdSource": row.get("patientIdSource", "source"),
                    "label": row["label"],
                    **({"pool": "training"} if split.version == 3 else {}),
                }
            )
        )
        + 1
        for patient, rows in all_groups.items()
        for row in rows
    )
    # Each membership carries its domain twice: as a field and inside planId.
    # Count both JSON-escaped copies conservatively before allocating any plans;
    # the baseline planId already counted above provides additional headroom.
    extra_domain_bytes = (
        sum(
            len(_json({"domain": value, "planId": f"seed:4294967295/domain:{value}"}))
            for value in held_out_domains
        )
        * row_count
        * len(split.seeds)
    )
    if row_bytes * plan_count + extra_domain_bytes > max_bytes:
        finding(
            "PROTOCOL_DOCUMENT_LIMIT",
            "The requested memberships exceed the 15 MiB protocol limit. Reduce plans or cohort size.",
        )
        return []
    if len(domains) != len(patients) and split.mode == "leave_one_domain_out":
        return []
    plans = []

    def add(seed, fold, train_pool, test, context, *, tune=None, explicit_val=None, **metadata):
        excluded_validation = set()
        if fixed_validation is not None:
            explicit_val = set(validation_groups)
            if "domain" in metadata:
                excluded_validation = {
                    patient
                    for patient, rows in validation_groups.items()
                    if any(
                        evaluator.field(row, split.domainField) == metadata["domain"]
                        for row in rows
                    )
                }
                explicit_val -= excluded_validation
                if excluded_validation:
                    finding(
                        "DOMAIN_VALIDATION_EXCLUDED",
                        f"Seed {seed}, {context}: {len(excluded_validation)} fixed validation groups belong to the held-out domain and are excluded from early stopping.",
                        "warning",
                    )
                    metadata["excludedValidation"] = {
                        "groups": len(excluded_validation),
                        "slides": sum(
                            len(validation_groups[patient]) for patient in excluded_validation
                        ),
                        "groupIds": sorted(excluded_validation),
                    }
        val = (
            explicit_val
            if explicit_val is not None
            else _subset(
                train_pool,
                split.validationFraction,
                seed,
                [context, "early_stop"],
                spec,
                groups,
                finding,
                role="early-stop validation",
            )
        )
        train = set(train_pool) - set(val)
        role_sets = {"train": train, "val": set(val), "test": set(test), "tune": set(tune or ())}
        if sum(map(len, role_sets.values())) != len(set().union(*role_sets.values())):
            finding(
                "PATIENT_PARTITION_LEAKAGE", "A group appears in multiple roles of the same plan."
            )
        assignment = {
            patient: role for role, members in role_sets.items() for patient in sorted(members)
        }
        expected = (patients - set(metadata.pop("excluded", ()))) | (
            set(validation_groups) - excluded_validation
        )
        if set(assignment) != expected:
            finding(
                "INCOMPLETE_ASSIGNMENTS",
                "Every eligible group needs one role in each applicable evaluation plan.",
            )
        plans.append(
            (
                {
                    "seed": seed,
                    "fold": fold,
                    "planId": f"seed:{seed}/{context}",
                    "phase": metadata.pop("phase", "evaluation"),
                    **metadata,
                },
                assignment,
            )
        )

    for seed in split.seeds:
        if split.mode == "kfold":
            folds = _folds(patients, split.folds, seed, "reported_test", spec, groups, finding)
            for fold, test in enumerate(folds or ()):
                add(seed, fold, patients - test, test, f"fold:{fold}")
        elif split.mode == "monte_carlo":
            for repeat in range(split.repeats):
                context = f"repeat:{repeat}"
                test = _subset(
                    patients,
                    split.testFraction,
                    seed,
                    [context, "reported_test"],
                    spec,
                    groups,
                    finding,
                    role="reported test",
                )
                add(seed, repeat, patients - test, test, context, repeat=repeat)
        elif split.mode == "leave_one_domain_out":
            for fold, domain in enumerate(held_out_domains):
                test = {patient for patient, value in domains.items() if value == domain}
                add(seed, fold, patients - test, test, f"domain:{domain}", domain=domain)
        elif split.mode == "nested_kfold":
            outer = _folds(patients, split.outerFolds, seed, "outer_test", spec, groups, finding)
            for outer_fold, test in enumerate(outer or ()):
                development = patients - test
                context = f"outer:{outer_fold}"
                inner = _folds(
                    development,
                    split.innerFolds,
                    seed,
                    [context, "inner_tune"],
                    spec,
                    groups,
                    finding,
                )
                for inner_fold, tune in enumerate(inner or ()):
                    add(
                        seed,
                        inner_fold,
                        development - tune,
                        set(),
                        f"{context}/inner:{inner_fold}",
                        tune=tune,
                        phase="inner",
                        outerFold=outer_fold,
                        innerFold=inner_fold,
                        excluded=test,
                    )
                add(
                    seed,
                    outer_fold,
                    development,
                    test,
                    context,
                    phase="outer",
                    outerFold=outer_fold,
                )
        else:
            explicit_val = None
            if split.heldOutSource == "fractions":
                test = _subset(
                    patients,
                    split.testFraction,
                    seed,
                    ["held_out", "reported_test"],
                    spec,
                    groups,
                    finding,
                    role="reported test",
                )
                train_pool = patients - test
            else:
                assignment = (
                    dict(fixed)
                    if split.heldOutSource == "rules"
                    else _imported_groups(spec, groups, evaluator, finding)
                )
                remaining = patients - set(assignment)
                if split.heldOutSource == "rules" and not split.rules.train:
                    assignment.update({patient: "train" for patient in remaining})
                elif remaining:
                    finding(
                        "UNASSIGNED_RULE_GROUPS"
                        if split.heldOutSource == "rules"
                        else "INVALID_IMPORTED_PARTITION",
                        "Some eligible groups have no partition. Assign them or leave training rules empty.",
                    )
                test = {patient for patient, role in assignment.items() if role == "test"}
                train_pool = {patient for patient, role in assignment.items() if role == "train"}
                val = {patient for patient, role in assignment.items() if role == "val"}
                has_explicit_val = (
                    bool(split.rules.val) if split.heldOutSource == "rules" else bool(val)
                )
                if has_explicit_val:
                    explicit_val = val
                    train_pool |= val
                if not test:
                    finding(
                        "MISSING_HELD_OUT_TEST",
                        "Held-out validation requires an explicit final test set.",
                    )
            add(seed, None, train_pool, test, "held_out", explicit_val=explicit_val)
    return plans


def check_modern_plan(spec, groups, metadata, assignment, finding):
    result = dict(metadata)
    for role in ROLES:
        patients = [patient for patient, assigned in assignment.items() if assigned == role]
        rows = [row for patient in patients for row in groups[patient]]
        classes = Counter(groups[patient][0]["label"] for patient in patients)
        result[role] = {
            "patients": sum(
                groups[patient][0].get("patientIdSource") != "slide_fallback"
                for patient in patients
            ),
            "groups": len(patients),
            "fallbackSlides": sum(row.get("patientIdSource") == "slide_fallback" for row in rows),
            "slides": len(rows),
            "classes": {label: classes[label] for label in spec.target.classes},
        }
        required = role in {"train", "val"} or role == (
            "tune" if metadata["phase"] == "inner" else "test"
        )
        if not required:
            continue
        if len(patients) < spec.constraints.minPatientsPerPartition:
            finding(
                "PARTITION_TOO_SMALL",
                f"{metadata['planId']}: {role} has fewer than {spec.constraints.minPatientsPerPartition} groups.",
            )
        for label in spec.target.classes:
            if classes[label] < spec.constraints.minPatientsPerClass:
                domain_test = (
                    role == "test"
                    and spec.split.mode == "leave_one_domain_out"
                    and metadata["phase"] != "final"
                )
                finding(
                    "DOMAIN_TEST_CLASS_IMBALANCE" if domain_test else "PARTITION_CLASS_TOO_SMALL",
                    f"{metadata['planId']}: {role} has fewer than {spec.constraints.minPatientsPerClass} groups in class '{label}'.",
                    "warning" if domain_test else "error",
                )
    return result


def modern_summary(spec, groups, plans):
    evaluations = [
        (metadata, assignment)
        for metadata, assignment in plans
        if metadata["phase"] not in {"inner", "final"}
    ]
    appearances = Counter(
        (metadata["seed"], patient)
        for metadata, assignment in evaluations
        for patient, role in assignment.items()
        if role == "test"
    )
    counts = [appearances[(seed, patient)] for seed in spec.split.seeds for patient in groups]
    return {
        "strategy": spec.split.mode,
        "splitVersion": spec.split.version,
        "evaluationPlanCount": len(evaluations),
        "innerPlanCount": sum(metadata["phase"] == "inner" for metadata, _assignment in plans),
        "oofCoverage": {
            "groups": len(groups),
            "testedGroups": len({patient for _seed, patient in appearances}),
            "minTestAppearances": min(counts, default=0),
            "maxTestAppearances": max(counts, default=0),
            "complete": bool(counts) and all(count == 1 for count in counts),
        },
        "roleDescriptions": {
            "train": "Model fitting",
            "val": "Early stopping only; the stopping criterion is configured in the MIL experiment",
            "test": "Reported evaluation; excluded from early stopping and model selection",
            "tune": "Inner held-out data for model selection; outer test groups never enter inner plans",
        },
        "validationFraction": spec.split.validationFraction,
        "validationFractionScope": "Fraction of each training pool after reported test and inner tuning data are removed; explicit validation assignments override it",
        "stratified": spec.split.stratify,
        "modelSelection": "Select using inner tuning results, then refit on outer development data with separate early stopping"
        if spec.split.mode == "nested_kfold"
        else None,
    }
