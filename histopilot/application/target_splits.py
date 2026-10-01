"""Fixed train/test construction, independent of model training and features."""

from __future__ import annotations

from collections import Counter, defaultdict

from pydantic import ValidationError

from histopilot.application.protocols import (
    CANONICAL,
    FilterEvaluator,
    FilterFailure,
    ProtocolService,
    cohort_statistics,
    development_selection_groups,
    fixed_assignments,
    forbidden_name,
    name_key,
    valid_patient,
)
from histopilot.schemas.protocols import FixedRules, SplitSpec, TargetSpec, iter_conditions
from histopilot.schemas.target_splits import TargetSplitPartitionPreviewRequest, TargetSplitSpec
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.io import (
    canonical_json,
    content_hash,
    read_json_bounded,
    utc_now,
    write_json_atomic,
)
from histopilot.storage.project_lock import (
    StorageError,
    ensure_managed_directory,
    fsync_directory,
    writer_lock,
)

ALGORITHM = "histopilot-target-training-testing-v2"
# A failed test-cohort step is kept beside, never inside, the content-addressed version.
TEST_COHORT_FAILURE = "test-cohort-failure.json"
MAX_FAILURE_BYTES = 64 * 1024
FAILURE_ERRORS = (StorageError, ValueError, KeyError, TypeError, OSError)
SLIDE_ALGORITHM = "histopilot-target-training-testing-slide-v1"


def _algorithm(spec):
    return SLIDE_ALGORITHM if spec.splitUnit == "slide" else ALGORITHM


def _selection_groups(rows, unit):
    if unit == "slide":
        return {(1, row["slideId"]): [row] for row in rows}
    return development_selection_groups(rows)


def _selection_stats(rows, unit):
    if unit != "slide":
        return cohort_statistics(rows)
    return {
        "totalSlides": len(rows),
        "patientCount": 0,
        "fallbackSlideCount": 0,
        "groupCount": len(rows),
        "unlinkedSlideCount": 0,
        "sample": [
            {
                "slideId": row["slideId"],
                "patientId": row.get("patientId"),
                **({"patientIdSource": row["patientIdSource"]} if "patientIdSource" in row else {}),
                "attributes": row["attributes"],
            }
            for row in rows[:5]
        ],
    }


def _percent(fraction):
    return f"{fraction * 100:.3g}%"


def _achieved_test_fraction(spec, assignments, stratified, finding):
    """The testing share a random split reached, in split units; warns beyond 5 points.

    Each stratum is rounded to whole units separately, and a stratum of two or more keeps
    at least one unit in each set, so small or stratified cohorts can miss the request.
    """
    total = len(assignments)
    if not total:
        return {}
    testing = sum(role == "test" for role in assignments.values())
    fraction, requested = testing / total, spec.split.testFraction
    units = "slides" if spec.splitUnit == "slide" else "patient groups"
    if abs(fraction - requested) * 100 > 5 + 1e-9:
        finding(
            "TEST_FRACTION_DIFFERS",
            f"Testing holds {testing} of {total} {units} ({_percent(fraction)}), not the "
            f"requested {_percent(requested)}. "
            + (
                f"Each value of {spec.split.stratifyField} is split on its own, rounded to whole "
                f"{units} with at least one in each set."
                if stratified
                else f"The testing set is rounded to whole {units} and keeps at least one in "
                "each set."
            ),
            "warning",
        )
    return {"testingUnits": testing, "splitUnits": total, "achievedTestFraction": fraction}


def _reference(document):
    return {key: document[key] for key in ("id", "contentHash")}


def _finding_collector(findings):
    def finding(code, message, severity="error"):
        item = {"code": code, "message": message, "severity": severity}
        if item not in findings:
            findings.append(item)

    return finding


def _target_source_identity(dataset, fields, field):
    """Identify the original table column, including separately imported tables."""
    column = fields.get(field, {})
    mapping = dataset["manifest"].get("provenance", {}).get("mapping", {})
    if "patientAttributes" in mapping:
        patient_fields = {item["key"] for item in mapping["patientAttributes"]}
        table = "patient" if field in patient_fields else "main"
    else:
        # Older/minimal dictionaries may lack the import mapping. Do not equate
        # equally named columns from different declared ownership/source tables.
        table = column.get("sourceTable", column.get("owner", "slide"))
    return table, column.get("sourceColumn", field)


def _testing_target_findings(dataset, fields, training, testing, finding):
    if training is None or testing is None:
        return
    if any(
        getattr(training, key) != getattr(testing, key)
        for key in ("task", "unit", "classes", "positiveClass")
    ):
        finding(
            "TARGET_CONTRACT_MISMATCH",
            "Testing targets must preserve the training task, prediction unit, class order, and positive class.",
        )
    if _target_source_identity(dataset, fields, training.field) == _target_source_identity(
        dataset, fields, testing.field
    ) and any(
        training.labels[raw] != testing.labels[raw]
        for raw in training.labels.keys() & testing.labels.keys()
    ):
        finding(
            "TARGET_LABEL_MAPPING_MISMATCH",
            "Training and testing aliases of the same source label column must preserve the same class mapping.",
        )


def _target_distribution(rows, field, target, *, include_patients=True):
    """Count slides and mutually exclusive verified-patient label categories.

    A patient belongs to a class only when every selected slide maps to that
    class. Mixed labels, missing values and unmapped values have separate buckets;
    acknowledged slide fallback groups never masquerade as verified patients.
    """
    evaluator = FilterEvaluator()
    values = Counter(evaluator.field(row, field) for row in rows)
    patients = defaultdict(list)
    for row in rows if include_patients else ():
        if valid_patient(row) and row.get("patientIdSource") != "slide_fallback":
            patients[row["patientId"]].append(row)
    uniform_values, mixed_values = Counter(), 0
    class_counts, patient_counts = Counter(), Counter()
    mixed, missing, unmapped, unlabeled = 0, 0, 0, 0
    for group in patients.values():
        raw_values = {evaluator.field(row, field) for row in group}
        if len(raw_values) == 1:
            uniform_values[next(iter(raw_values))] += 1
        else:
            mixed_values += 1
        if target is None:
            unlabeled += 1
            continue
        mapped = {
            target.labels[value]
            for value in raw_values
            if value is not None and value in target.labels
        }
        if len(mapped) > 1:
            mixed += 1
        elif None in raw_values:
            missing += 1
        elif any(value not in target.labels for value in raw_values):
            unmapped += 1
        elif mapped:
            patient_counts[next(iter(mapped))] += 1
    if target:
        class_counts.update(
            target.labels[value]
            for value, count in values.items()
            if value is not None and value in target.labels
            for _ in range(count)
        )
    return {
        "field": field,
        "values": [
            {"value": value, "slides": count, "patients": uniform_values[value]}
            for value, count in sorted(
                values.items(), key=lambda item: (item[0] is not None, item[0] or "")
            )
        ],
        "distinctCount": len(values.keys() - {None}),
        "mixedValuePatients": mixed_values,
        "classCounts": {label: class_counts[label] for label in target.classes} if target else {},
        "patientClassCounts": {label: patient_counts[label] for label in target.classes}
        if target and include_patients
        else {},
        "missingSlides": values[None],
        "unmappedSlides": sum(
            count
            for value, count in values.items()
            if value is not None and target and value not in target.labels
        ),
        "mixedLabelPatients": mixed,
        "missingLabelPatients": missing,
        "unmappedLabelPatients": unmapped,
        "labeledPatients": sum(patient_counts.values()),
        "unlabeledPatients": unlabeled + missing + unmapped,
    }


class _PreviewStore:
    """Present an in-memory construction intent without leaving transient drafts."""

    def __init__(self, store, spec):
        self.store, self.spec = store, spec

    def __getattr__(self, name):
        return getattr(self.store, name)

    def get_draft(self, _identity):
        return {
            "kind": "experiment",
            "status": "editable",
            "revision": 1,
            "payload": {"type": "analysis-protocol", "spec": self.spec},
        }


class TargetSplitService:
    def __init__(self, store, filesystem=None):
        self.store = store
        self.filesystem = filesystem or LocalFilesystem(())
        self.protocols = ProtocolService(store, self.filesystem)

    def _draft_spec(self, draft_id, revision):
        draft = self.store.get_draft(draft_id)
        if type(revision) is not int or revision < 1 or draft["revision"] != revision:
            raise StorageError("The target/split draft changed. Reload it.", "REVISION_CONFLICT")
        if draft["status"] != "editable":
            raise StorageError("This target/split is frozen. Copy it to edit.", "DRAFT_FROZEN")
        payload = draft["payload"]
        if (
            draft["kind"] != "experiment"
            or not isinstance(payload, dict)
            or set(payload) != {"type", "spec"}
            or payload["type"] != "target-split"
        ):
            raise StorageError("Select a target/split draft.", "INVALID_TARGET_SPLIT_DRAFT", 422)
        try:
            return TargetSplitSpec.model_validate(payload["spec"])
        except ValidationError as error:
            details = "; ".join(item["msg"] for item in error.errors(include_input=False)[:8])
            raise StorageError(
                f"Invalid target/split: {details}", "INVALID_TARGET_SPLIT_SPEC", 422
            ) from error

    def preview(self, draft_id, expected_revision):
        return self.preview_spec(self._draft_spec(draft_id, expected_revision))

    def _partition_selection(self, spec, finding):
        """Select fixed raw-record partitions before any target field is read."""
        dataset, fields, rows = self.protocols._load_dataset(spec.datasetId)
        conditions = [*spec.eligibility, *spec.split.trainRules, *spec.split.testRules]
        used = [item.field for item in iter_conditions(conditions)]
        if spec.split.partitionField:
            used.append(spec.split.partitionField)
        stratify_field = spec.split.stratifyField if spec.split.stratify else None
        if spec.split.method == "random" and spec.split.stratify and not stratify_field:
            finding(
                "STRATIFICATION_FIELD_REQUIRED",
                "Choose a metadata field for stratification, or turn stratification off. "
                "Partitioning no longer uses prediction targets selected later.",
            )
        if stratify_field and spec.split.method == "random":
            used.append(stratify_field)
            source = fields.get(stratify_field, {}).get("sourceColumn", stratify_field)
            if stratify_field in CANONICAL or forbidden_name(source, target=True):
                finding(
                    "IDENTIFIER_STRATIFICATION",
                    "Choose a metadata attribute rather than an identifier for stratification.",
                )
        for field in sorted(set(used)):
            if field not in fields and field not in CANONICAL:
                finding("UNKNOWN_FIELD", f"Field '{field}' is not in the dataset dictionary.")
        if len({row["slideId"] for row in rows}) != len(rows):
            finding("DUPLICATE_SLIDE_ID", "Dataset slide identifiers must be unique.")
        evaluator = FilterEvaluator()
        eligible, assignments, achieved = [], {}, {}
        groups = {}
        direct = {role: [] for role in ("train", "test")}
        expanded = {role: [] for role in ("train", "test")}
        try:
            evaluator.prepare(conditions)
            eligible = [row for row in rows if evaluator.conjunction(row, spec.eligibility)]
            groups = _selection_groups(eligible, spec.splitUnit)
            if spec.split.method == "rules":

                def rule_finding(code, message, severity="error"):
                    if spec.splitUnit == "slide" and code == "OVERLAPPING_PATIENT_RULES":
                        code = "OVERLAPPING_SLIDE_RULES"
                        message = "The same slide matches both training and testing conditions."
                    finding(code, message, severity)

                assignments, direct, expanded = fixed_assignments(
                    groups,
                    FixedRules(train=spec.split.trainRules, test=spec.split.testRules),
                    evaluator,
                    rule_finding,
                )
                # At most one side takes the eligible remainder. Testing takes it
                # only on request, which needs training defined by its own rules.
                if spec.split.testRemaining and not spec.split.trainRules:
                    finding(
                        "TESTING_REMAINDER_NEEDS_TRAINING_RULES",
                        "Testing uses every eligible "
                        + ("slide" if spec.splitUnit == "slide" else "patient group")
                        + " outside training. Add at least one training condition.",
                    )
                elif spec.split.testRemaining or not spec.split.trainRules:
                    remainder = "test" if spec.split.testRemaining else "train"
                    for patient in groups:
                        assignments.setdefault(patient, remainder)
            elif spec.split.method == "imported":
                roles = {
                    **dict.fromkeys(spec.split.trainValues, "train"),
                    **dict.fromkeys(spec.split.testValues, "test"),
                }
                for patient, group in groups.items():
                    matches = set()
                    for role in ("train", "test"):
                        matched = [
                            row
                            for row in group
                            if roles.get(evaluator.field(row, spec.split.partitionField)) == role
                        ]
                        direct[role].extend(matched)
                        if matched:
                            matches.add(role)
                            expanded[role].extend(group)
                    if len(matches) > 1:
                        finding(
                            "IMPORTED_PATIENT_LEAKAGE",
                            "Predefined values assign one patient's slides to both training and testing.",
                        )
                    elif matches:
                        assignments[patient] = next(iter(matches))
            else:
                strata = defaultdict(list)
                for patient, group in groups.items():
                    values = (
                        tuple(
                            sorted(
                                {evaluator.field(row, stratify_field) for row in group},
                                key=lambda value: (value is not None, value or ""),
                            )
                        )
                        if stratify_field
                        else ()
                    )
                    strata[values].append(patient)
                for values, patients in sorted(
                    strata.items(),
                    key=lambda item: canonical_json(item[0], ascii=True, compact=True),
                ):
                    ordered = sorted(
                        patients,
                        key=lambda patient: (
                            content_hash([_algorithm(spec), spec.split.seed, values, patient]),
                            patient,
                        ),
                    )
                    count = int(len(ordered) * spec.split.testFraction + 0.5)
                    if spec.split.testFraction and len(ordered) > 1:
                        count = max(1, min(len(ordered) - 1, count))
                    for index, patient in enumerate(ordered):
                        assignments[patient] = "test" if index < count else "train"
                achieved = _achieved_test_fraction(spec, assignments, bool(stratify_field), finding)
        except FilterFailure as error:
            finding(error.code, str(error))
        partitions = {
            role: sorted(
                [
                    row
                    for patient, group in groups.items()
                    if assignments.get(patient) == role
                    for row in group
                ],
                key=lambda row: (row.get("patientId") or "", row["slideId"]),
            )
            for role in ("train", "test")
        }
        selected = partitions["train"] + partitions["test"]
        if spec.splitUnit == "patient":
            self.protocols._identity_findings(
                selected, development_selection_groups(selected), finding
            )
        if not partitions["train"]:
            finding(
                "EMPTY_TRAINING_SET",
                "Select at least one slide for training."
                if spec.splitUnit == "slide"
                else "Select at least one patient group for training.",
            )
        if not partitions["test"]:
            finding(
                "NO_TESTING_SET",
                "No testing slides were reserved. Independent evaluation can use a separate cohort later."
                if spec.splitUnit == "slide"
                else "No testing patients were reserved. Independent evaluation can use a separate cohort later.",
                "warning",
            )
        # Check raw selected slides, including those later excluded by label rules.
        from histopilot.application.evaluations import duplicate_test_sources, slide_sources

        training_ids = {row["slideId"] for row in partitions["train"]}
        testing_ids = {row["slideId"] for row in partitions["test"]}
        sources = slide_sources(self.store, dataset, selected, training_ids | testing_ids)
        duplicate_test_sources(
            {slide: identities for slide, identities in sources.items() if slide in testing_ids},
            finding,
        )
        if spec.splitUnit == "slide":
            duplicate_test_sources(
                {
                    slide: identities
                    for slide, identities in sources.items()
                    if slide in training_ids
                },
                lambda _code, message: finding("DUPLICATE_TRAINING_SLIDE_SOURCE", message),
            )
        training_sources = {
            identity for slide in training_ids for identity in sources.get(slide, ())
        }
        if any(training_sources & sources.get(slide, set()) for slide in testing_ids):
            finding(
                "TRAIN_TEST_SOURCE_OVERLAP",
                "Training and testing contain the same physical slide under different identifiers.",
            )
        selection = {}
        for role, assigned in partitions.items():
            mode = spec.split.method
            if mode == "rules" and not getattr(spec.split, f"{role}Rules"):
                remainder = (
                    ("test" if spec.split.trainRules else None)
                    if spec.split.testRemaining
                    else "train"
                )
                mode = "remaining" if role == remainder else "none"
            # Random assignment and the remaining-training pool do not apply a
            # row-level rule. Their selected rows already include whole groups.
            if mode in {"random", "remaining", "none"}:
                direct[role] = expanded[role] = assigned
            selection[role] = {
                "mode": mode,
                "directMatches": _selection_stats(
                    sorted(direct[role], key=lambda row: row["slideId"]), spec.splitUnit
                ),
                "expanded": _selection_stats(
                    sorted(expanded[role], key=lambda row: row["slideId"]), spec.splitUnit
                ),
                "assigned": _selection_stats(assigned, spec.splitUnit),
            }
        return dataset, fields, rows, eligible, partitions, selection, achieved

    @staticmethod
    def _selection_summary(rows, eligible, partitions, unit="patient", achieved=None):
        training, testing = partitions["train"], partitions["test"]
        selected = training + testing
        return {
            # Random splits: testing and all split units (slides or patient groups), and
            # the fraction in testing, which rounding can move off the requested one.
            **(achieved or {}),
            "totalSlides": len(rows),
            "eligibleSlides": len(eligible),
            "selectedSlides": len(selected),
            "excludedSlides": len(rows) - len(selected),
            "eligibilityExcludedSlides": len(rows) - len(eligible),
            "partitionExcludedSlides": len(eligible) - len(selected),
            "trainingSlides": len(training),
            "testingSlides": len(testing),
            "trainingPatients": _selection_stats(training, unit)["patientCount"],
            "testingPatients": _selection_stats(testing, unit)["patientCount"],
            "trainingGroups": _selection_stats(training, unit)["groupCount"],
            "testingGroups": _selection_stats(testing, unit)["groupCount"],
        }

    @staticmethod
    def _partition_detail(rows, field=None, target=None, unit="patient"):
        stats = _selection_stats(rows, unit)
        result = {
            "slides": len(rows),
            "patients": stats["patientCount"],
            "groups": stats["groupCount"],
            "fallbackSlides": stats["fallbackSlideCount"],
            "unlinkedSlides": stats["unlinkedSlideCount"],
        }
        if field is not None:
            result["target"] = _target_distribution(
                rows, field, target, include_patients=unit != "slide"
            )
        return result

    def partition_preview(self, request):
        request = TargetSplitPartitionPreviewRequest.model_validate(request)
        findings = []
        finding = _finding_collector(findings)
        dataset, fields, rows, eligible, partitions, selection, achieved = (
            self._partition_selection(request, finding)
        )
        allocation_valid = not any(item["severity"] == "error" for item in findings)
        targets = {}
        raw_targets = {
            "train": request.target,
            "test": request.testTarget
            if "testTarget" in request.model_fields_set
            else request.target,
        }
        for role, raw in raw_targets.items():
            if raw is not None:
                try:
                    targets[role] = TargetSpec.model_validate(raw)
                except ValidationError as error:
                    details = "; ".join(
                        item["msg"] for item in error.errors(include_input=False)[:3]
                    )
                    finding("INVALID_TARGET_MAPPING", f"{role.title()} target: {details}")
                    continue
                if role == "train" and targets[role].keeps_unlabeled:
                    finding(
                        "TRAINING_LABELS_REQUIRED",
                        "Training targets need a label for every slide. Exclude or block "
                        "missing and unmapped values; only testing can keep slides unlabeled.",
                    )
        _testing_target_findings(
            dataset, fields, targets.get("train"), targets.get("test"), finding
        )
        details = {}
        for role in ("train", "test"):
            target = targets.get(role)
            if (
                target is not None
                and "splitUnit" in request.model_fields_set
                and target.unit != request.splitUnit
            ):
                finding(
                    "SPLIT_TARGET_UNIT_MISMATCH",
                    "The target prediction unit must match the selected split unit.",
                )
            field = target.field if target else getattr(request.targetFields, role)
            # Explicit inference never consults a label column, including stale UI selections.
            if (
                role == "test"
                and "testTarget" in request.model_fields_set
                and request.testTarget is None
            ):
                field = None
            if field and field not in fields and field not in CANONICAL:
                finding("UNKNOWN_FIELD", f"Field '{field}' is not in the dataset dictionary.")
            details[role] = {
                **self._partition_detail(partitions[role], field, target, request.splitUnit),
                "selection": selection[role],
            }
        return {
            "algorithm": _algorithm(request),
            "dataset": cohort_statistics(rows),
            "cohort": cohort_statistics(eligible),
            "summary": {
                **self._selection_summary(rows, eligible, partitions, request.splitUnit, achieved),
                **(
                    {"splitUnit": request.splitUnit}
                    if "splitUnit" in request.model_fields_set
                    else {}
                ),
            },
            "partitions": details,
            "findings": findings,
            "valid": not any(item["severity"] == "error" for item in findings),
            "membershipStatus": "fixed" if allocation_valid else "provisional",
        }

    def preview_spec(self, spec):
        spec = TargetSplitSpec.model_validate(spec)
        findings = []
        finding = _finding_collector(findings)
        dataset, fields, rows, eligible, partitions, _selection, achieved = (
            self._partition_selection(spec, finding)
        )
        selection = self._selection_summary(rows, eligible, partitions, spec.splitUnit, achieved)
        test_target = spec.testTarget if "testTarget" in spec.model_fields_set else spec.target
        targets = {"train": spec.target, "test": test_target}
        _testing_target_findings(dataset, fields, spec.target, test_target, finding)
        mapping = dataset["manifest"].get("provenance", {}).get("mapping", {})
        identifiers = {
            name_key(mapping[key])
            for key in (
                "slideIdColumn",
                "patientIdColumn",
                "patientSourceSlideIdColumn",
                "patientSourcePatientIdColumn",
            )
            if isinstance(mapping.get(key), str)
        }
        partition_source = fields.get(spec.split.partitionField, {}).get(
            "sourceColumn", spec.split.partitionField
        )
        for target in targets.values():
            if target is None:
                continue
            source = fields.get(target.field, {}).get("sourceColumn", target.field)
            if target.field not in fields and target.field not in CANONICAL:
                finding(
                    "UNKNOWN_FIELD", f"Field '{target.field}' is not in the dataset dictionary."
                )
            if (
                target.field in CANONICAL
                or forbidden_name(target.field, target=True)
                or forbidden_name(source, target=True)
                or name_key(source) in identifiers
            ):
                finding(
                    "IDENTIFIER_TARGET",
                    "Identifiers and partition fields cannot serve as target labels.",
                )
            if partition_source and name_key(source) == name_key(partition_source):
                finding("SPLIT_TARGET_LEAKAGE", "The partition column cannot also be the target.")
        evaluator = FilterEvaluator()
        included, role_exclusions, unlabeled_testing = {}, {}, 0
        for role, selected in partitions.items():
            target = targets[role]
            included[role], role_exclusions[role] = [], Counter()
            for row in selected:
                label = None
                if target is not None:
                    raw = evaluator.field(row, target.field)
                    label = target.labels.get(raw) if raw is not None else None
                    if label is None:
                        missing = raw is None
                        policy = target.missing if missing else target.unmapped
                        # Testing may keep a slide without a label: it is predicted, never
                        # scored. The schema refuses this policy on training targets.
                        if policy == "unlabeled" and role == "test":
                            unlabeled_testing += 1
                            included[role].append({**row, "label": None, "partition": role})
                            continue
                        role_exclusions[role]["missingLabel" if missing else "unmappedLabel"] += 1
                        if policy == "block":
                            finding(
                                "MISSING_LABEL" if missing else "UNMAPPED_LABEL",
                                f"{role.title()} slides have missing or unmapped target labels. "
                                "Map or explicitly exclude them.",
                            )
                        continue
                included[role].append({**row, "label": label, "partition": role})
            if (
                target
                and target.unit == "patient"
                and any(
                    len({row["label"] for row in group if row["label"] is not None}) > 1
                    for group in development_selection_groups(included[role]).values()
                )
            ):
                finding(
                    "MIXED_PATIENT_LABELS",
                    f"A {role} patient has conflicting target labels; no majority label is selected.",
                )
        if unlabeled_testing:
            finding(
                "UNLABELED_TESTING_SLIDES",
                f"{unlabeled_testing} testing slides have no label for the testing target. "
                "They stay in the testing set and receive predictions, but no metric counts them.",
                "info",
            )
        training, testing = included["train"], included["test"]
        exclusions = role_exclusions["train"] + role_exclusions["test"]
        if exclusions:
            finding(
                "LABEL_EXCLUSIONS",
                f"{sum(exclusions.values())} slides are excluded by their target labels.",
                "warning",
            )
        if not training:
            finding("EMPTY_TRAINING_SET", "Training needs labeled slides after target exclusions.")
        train_counts = Counter(row["label"] for row in training)
        test_counts = Counter(row["label"] for row in testing if row["label"] is not None)
        for label in spec.target.classes:
            if not train_counts[label]:
                finding("TRAINING_CLASS_ABSENT", f"Training contains no '{label}' target labels.")
            if testing and test_target and not test_counts[label]:
                finding(
                    "TESTING_CLASS_ABSENT",
                    f"Testing contains no '{label}' target labels; some metrics will be unavailable.",
                    "warning",
                )
        if not testing and partitions["test"]:
            finding(
                "EMPTY_LABELED_TESTING_SET",
                "Target exclusions removed every testing slide. Review the testing target or select unlabeled inference.",
            )
        memberships = [
            {
                key: row.get(key)
                for key in ("slideId", "patientId", "patientIdSource", "label", "partition")
            }
            for row in training + testing
        ]
        memberships.sort(key=lambda row: (row["partition"], row["patientId"] or "", row["slideId"]))
        details = {
            role: self._partition_detail(
                selected,
                targets[role].field if targets[role] else None,
                targets[role],
                spec.splitUnit,
            )
            for role, selected in partitions.items()
        }
        all_stats = _selection_stats(training + testing, spec.splitUnit)
        summary = {
            **selection,
            "selectedTrainingSlides": selection["trainingSlides"],
            "selectedTestingSlides": selection["testingSlides"],
            "selectedTrainingPatients": selection["trainingPatients"],
            "selectedTestingPatients": selection["testingPatients"],
            "includedSlides": len(memberships),
            "excludedSlides": len(rows) - len(memberships),
            "includedPatients": all_stats["patientCount"],
            "trainingSlides": len(training),
            "testingSlides": len(testing),
            "trainingPatients": _selection_stats(training, spec.splitUnit)["patientCount"],
            "testingPatients": _selection_stats(testing, spec.splitUnit)["patientCount"],
            "trainingGroups": _selection_stats(training, spec.splitUnit)["groupCount"],
            "testingGroups": _selection_stats(testing, spec.splitUnit)["groupCount"],
            "classCounts": {
                label: train_counts[label] + test_counts[label] for label in spec.target.classes
            },
            "trainingClassCounts": {label: train_counts[label] for label in spec.target.classes},
            "testingClassCounts": {label: test_counts[label] for label in test_target.classes}
            if test_target
            else {},
            # Only versions that keep unlabeled testing slides carry this count, so earlier
            # frozen summaries keep their exact content.
            **({"testingUnlabeledSlides": unlabeled_testing} if unlabeled_testing else {}),
            "trainingPatientClassCounts": _target_distribution(
                training, spec.target.field, spec.target, include_patients=spec.splitUnit != "slide"
            )["patientClassCounts"],
            "testingPatientClassCounts": _target_distribution(
                testing, test_target.field, test_target, include_patients=spec.splitUnit != "slide"
            )["patientClassCounts"]
            if test_target
            else {},
            "labelExclusions": dict(exclusions),
            "trainingLabelExclusions": dict(role_exclusions["train"]),
            "testingLabelExclusions": dict(role_exclusions["test"]),
            "targetUnit": spec.target.unit,
            "testingPurpose": "independent" if test_target else "inference",
            **({"splitUnit": spec.splitUnit} if "splitUnit" in spec.model_fields_set else {}),
            "grouping": "slide"
            if spec.splitUnit == "slide"
            else "patient_with_slide_fallback"
            if all_stats["fallbackSlideCount"]
            else "patient",
        }
        can_freeze = not any(item["severity"] == "error" for item in findings)
        result = {
            "spec": spec.model_dump(mode="json"),
            "dataset": _reference(dataset),
            "algorithm": _algorithm(spec),
            "summary": summary,
            "partitions": details,
            "memberships": memberships if can_freeze else [],
            "findings": findings,
            "canFreeze": can_freeze,
            "executionEnabled": False,
        }
        return {
            **result,
            "previewHash": content_hash(
                {
                    key: value
                    for key, value in result.items()
                    if key not in {"findings", "canFreeze", "executionEnabled"}
                }
            ),
        }

    def _document(self, identity):
        document = self.store.get_configuration(identity)
        if document["manifest"].get("kind") != "target-split":
            raise StorageError(
                "Select a frozen Targets & Splits version.", "INVALID_TARGET_SPLIT", 422
            )
        return document

    @staticmethod
    def _testing_operation(target_split_id):
        return "target-testing-" + target_split_id.removeprefix("configuration-")

    def get(self, identity):
        """Side-effect free. Freezing derives the testing cohort; older versions create it
        explicitly through ``create_test_cohort``. ``testCohort.required`` with no ``id``
        means the testing set has no derived labeled or unlabeled cohort yet."""
        document = self._document(identity)
        required = any(row["partition"] == "test" for row in document["manifest"]["memberships"])
        cohort_id = (
            self.store.published_configuration_id(self._testing_operation(identity))
            if required
            else None
        )
        state = (
            self.store.lifecycle.read()["records"]
            .get(f"configuration:{cohort_id}", {})
            .get("state", "active")
            if cohort_id
            else None
        )
        failure = self._test_cohort_failure(identity) if required and not cohort_id else None
        test_cohort = {"required": required, "id": cohort_id, "state": state}
        if failure:
            test_cohort.update(state="failed", error=failure)
        return {
            **document,
            "evaluationCohortId": cohort_id if state != "trashed" else None,
            "testCohort": test_cohort,
        }

    def _test_cohort_failure_path(self, target_split_id):
        # ``target_split_id`` names a stored configuration, never a caller-supplied path.
        return self.store.folder / "target-splits" / target_split_id / TEST_COHORT_FAILURE

    def _test_cohort_failure(self, target_split_id):
        """The error of the last attempt to derive the testing cohort, until one succeeds."""
        path = self._test_cohort_failure_path(target_split_id)
        try:
            if not path.exists():
                return None
            error = read_json_bounded(path, MAX_FAILURE_BYTES)["error"]
            return {"code": str(error["code"]), "message": str(error["message"])}
        except (StorageError, OSError, KeyError, TypeError):
            return None

    def _record_test_cohort_failure(self, target_split_id, error):
        """Keep a failed attempt so a reload still offers the retry. Never raises: the
        caller reports the original error."""
        path = self._test_cohort_failure_path(target_split_id)
        record = {
            "state": "failed",
            "at": utc_now(),
            "error": {
                "code": getattr(error, "code", "TARGET_TESTING_FAILED"),
                "message": str(error)[:4000],
            },
        }
        try:
            with writer_lock(self.store.folder, timeout=2):
                ensure_managed_directory(path.parent)
                write_json_atomic(path, record, limit=MAX_FAILURE_BYTES)
        except (StorageError, OSError):
            pass

    def _clear_test_cohort_failure(self, target_split_id):
        path = self._test_cohort_failure_path(target_split_id)
        try:
            if path.exists():
                with writer_lock(self.store.folder, timeout=2):
                    path.unlink(missing_ok=True)
                    fsync_directory(path.parent)
        except (StorageError, OSError):
            pass  # A stale record is ignored once the cohort exists.

    def freeze(
        self, draft_id, expected_revision, preview_hash, operation_id, *, version_label=None
    ):
        prior = self.store.configuration_publication(operation_id)
        if prior:
            manifest = prior["manifest"]
            if (
                manifest.get("kind") != "target-split"
                or manifest.get("previewHash") != preview_hash
            ):
                raise StorageError(
                    "This operation belongs to another target/split freeze.", "OPERATION_CONFLICT"
                )
        else:
            preview = self.preview(draft_id, expected_revision)
            if preview["previewHash"] != preview_hash:
                raise StorageError("Targets or partitions changed. Preview again.", "PREVIEW_STALE")
            if not preview["canFreeze"]:
                raise StorageError(
                    "Resolve blocking target/split findings before freezing.",
                    "TARGET_SPLIT_BLOCKED",
                    422,
                    findings=preview["findings"],
                )
            manifest = {
                "kind": "target-split",
                "schemaVersion": 2,
                "datasetId": preview["spec"]["datasetId"],
                **preview,
            }
        document = self.store.publish_configuration(
            draft_id,
            expected_revision=expected_revision,
            manifest=manifest,
            operation_id=operation_id,
            version_label=version_label,
        )
        try:
            self.create_test_cohort(document["id"])
        except FAILURE_ERRORS as error:
            # The version is frozen whatever happens here. Report the testing cohort as a
            # warning with an explicit retry, never as a failed (and re-submitted) freeze.
            # create_test_cohort saved the failure, so a later read offers the retry too.
            return {
                **self.get(document["id"]),
                "testCohortError": {
                    "code": getattr(error, "code", "TARGET_TESTING_FAILED"),
                    "message": str(error),
                },
            }
        return self.get(document["id"])

    def derive_protocol(self, target_split_id, training_split):
        source = self._document(target_split_id)
        split = SplitSpec.model_validate(training_split)
        if split.version != 4:
            raise StorageError(
                "Training design must use a development-only split.", "INVALID_TRAINING_SPLIT", 422
            )
        source_spec = source["manifest"]["spec"]
        # Every frozen spec stores "predictors": [] (the schema rejects others). The key stays in
        # the derived spec because it feeds the operation hash that makes derivation idempotent.
        spec = {key: source_spec[key] for key in ("datasetId", "target", "predictors")}
        spec.update(sourceTargetSplitId=target_split_id, split=split.model_dump(mode="json"))
        if "splitUnit" in source_spec:
            spec["splitUnit"] = source_spec["splitUnit"]
        operation = "target-training-" + content_hash(spec)
        prior = self.store.configuration_publication(operation)
        if prior:
            return prior
        preview = ProtocolService(_PreviewStore(self.store, spec), self.filesystem).preview(
            "derived", 1
        )
        if not preview["canFreeze"]:
            messages = "; ".join(
                item["message"] for item in preview["findings"] if item["severity"] == "error"
            )
            raise StorageError(
                f"Training design is not feasible: {messages}",
                "TRAINING_SPLIT_BLOCKED",
                422,
                findings=preview["findings"],
            )
        manifest = {
            "kind": "protocol",
            "datasetId": spec["datasetId"],
            "sourceTargetSplit": _reference(source),
            **{
                key: preview[key]
                for key in (
                    "spec",
                    "summary",
                    "memberships",
                    "partitions",
                    "findings",
                    "previewHash",
                    "executionEnabled",
                )
            },
            "algorithm": preview["summary"]["algorithm"],
        }
        return self.store.publish_configuration(manifest=manifest, operation_id=operation)

    def create_test_cohort(self, target_split_id):
        """Derive the testing set's cohort once; a failure is kept until a retry succeeds."""
        source = self._document(target_split_id)
        if not any(row["partition"] == "test" for row in source["manifest"]["memberships"]):
            return None
        operation = self._testing_operation(target_split_id)
        prior = self.store.configuration_publication(operation)
        if prior:
            return prior
        try:
            cohort = self._derive_test_cohort(source, target_split_id, operation)
        except FAILURE_ERRORS as error:
            self._record_test_cohort_failure(target_split_id, error)
            raise
        self._clear_test_cohort_failure(target_split_id)
        return cohort

    def _derive_test_cohort(self, source, target_split_id, operation):
        from histopilot.application.evaluations import EvaluationService
        from histopilot.schemas.evaluations import EvaluationSpec

        source_spec = source["manifest"]["spec"]
        target = source_spec.get("testTarget", source_spec["target"])
        spec = EvaluationSpec(
            datasetId=source["manifest"]["datasetId"],
            target=target,
            purpose="independent" if target else "inference",
            sourceTargetSplitId=target_split_id,
            **({"splitUnit": source_spec["splitUnit"]} if "splitUnit" in source_spec else {}),
        )
        preview, _guards = EvaluationService(self.store, self.filesystem)._prepare(spec)
        if not preview["canFreeze"]:
            messages = "; ".join(
                item["message"] for item in preview["findings"] if item["severity"] == "error"
            )
            raise StorageError(
                f"The testing cohort cannot be prepared: {messages}",
                "TARGET_TESTING_BLOCKED",
                422,
                findings=preview["findings"],
            )
        manifest = {
            "kind": "evaluation-cohort",
            "schemaVersion": 2,
            "datasetId": spec.datasetId,
            **preview,
        }
        source_tag = source.get("versionLabel", {}).get("tag") or "Targets & Splits"
        return self.store.publish_configuration(
            manifest=manifest,
            operation_id=operation,
            version_label={
                "tag": f"{source_tag[:52]} · Testing {target_split_id[-12:]}",
                "note": "Testing slides reserved by the frozen Targets & Splits version."
                if source_spec.get("splitUnit") == "slide"
                else "Testing patients reserved by the frozen Targets & Splits version.",
            },
        )
