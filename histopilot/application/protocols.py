"""Explicit targets, patient-grouped splits and immutable protocol publication.

New protocols hold development-only (version 4) splits. Frozen protocols of
split versions 1-3 stay readable, but their generators are gone. A dataset may
explicitly record acknowledged slide-ID fallback groups. Labels are never
dropped implicitly and a majority label is never selected.
"""

import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation

import regex
from pydantic import ValidationError

from histopilot.application.development_splits import (
    ALGORITHM_V4,
    ALGORITHM_V4_MIXED,
    development_assignments,
    development_summary,
    pool_counts,
    select_development_pools,
)
from histopilot.application.modern_splits import (
    check_modern_plan,
    group_class_counts,
    modern_summary,
)
from histopilot.schemas.protocols import (
    Condition,
    ConditionGroup,
    PoolSpec,
    ProtocolExploreRequest,
    ProtocolSpec,
    iter_conditions,
)
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore

MAX_RECORDS = 50000
MAX_MEMBERSHIPS = 500000
MAX_PROTOCOL_BYTES = 15 * 1024 * 1024
REGEX_TIMEOUT_SECONDS = 0.02
REGEX_TOTAL_SECONDS = 1.0
PARTITIONS = ("train", "val", "test")
CANONICAL = {
    "Slide_ID": "slideId",
    "slideId": "slideId",
    "Patient_ID": "patientId",
    "patientId": "patientId",
    "Slide_Path": "slidePath",
    "slidePath": "slidePath",
}


def _json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _serialized_spec(spec):
    value = spec.model_dump(mode="json")
    if spec.sourceTargetSplitId is None:
        value.pop("sourceTargetSplitId", None)
    return value


def pack_binding_snapshot(artifact: dict) -> dict:
    """Pin the selected physical representation without copying its large inventory."""
    return {
        key: artifact[key]
        for key in (
            "id",
            "materializationId",
            "featureSetId",
            "outputPath",
            "outputDtype",
            "sourceContentHash",
            "verification",
        )
        if key in artifact
    }


def _failure(message, code, status=409):
    return StorageError(message, code, status)


def name_key(value):
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def forbidden_name(value, *, target=False):
    name = name_key(value)
    if target and name in {"label", "labels", "target", "outcome", "y", "ytrue", "groundtruth"}:
        return False
    if name in {
        "id",
        "deid",
        "slideid",
        "patientid",
        "specimenid",
        "case",
        "caseid",
        "filename",
        "filepath",
        "slidepath",
        "patient",
        "slide",
        "specimen",
        "label",
        "labels",
        "target",
        "outcome",
        "partition",
        "split",
        "fold",
        "y",
        "ytrue",
        "groundtruth",
    }:
        return True
    return bool(
        re.fullmatch(r"(?:kfold|fold|split|partition)s?(?:id|index|assignment|label|[0-9]+)?", name)
    )


class FilterFailure(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class FilterEvaluator:
    """Typed scalar comparisons with bounded regular-expression execution."""

    def __init__(self):
        self.compiled = {}
        self.regex_seconds = 0.0

    def prepare(self, conditions):
        # Validate patterns even when there are no rows to evaluate.
        for condition in iter_conditions(conditions):
            if condition.op == "regex" and condition.value not in self.compiled:
                try:
                    self.compiled[condition.value] = regex.compile(condition.value)
                except regex.error as error:
                    raise FilterFailure(
                        "INVALID_REGEX", "A regular expression is invalid."
                    ) from error

    @staticmethod
    def field(row, key):
        return row.get(CANONICAL[key]) if key in CANONICAL else row["attributes"].get(key)

    @staticmethod
    def number(value):
        try:
            result = Decimal(str(value))
            if not result.is_finite():
                raise InvalidOperation
            return result
        except InvalidOperation as error:
            raise FilterFailure(
                "INVALID_FILTER_VALUE", "A numeric filter encountered a nonnumeric value."
            ) from error

    def equal(self, raw, expected):
        if expected is None or raw is None:
            return raw is expected
        if type(expected) is bool:
            values = {"true": True, "false": False, "1": True, "0": False}
            normalized = str(raw).casefold()
            if normalized not in values:
                raise FilterFailure(
                    "INVALID_FILTER_VALUE", "A boolean filter encountered an invalid value."
                )
            return values[normalized] is expected
        if type(expected) in (int, float):
            return self.number(raw) == self.number(expected)
        return raw == expected

    def matches(self, row, condition: Condition):
        if isinstance(condition, ConditionGroup):
            results = [self.matches(row, child) for child in condition.conditions]
            return all(results) if condition.op == "all" else any(results)
        raw = self.field(row, condition.field)
        op, value = condition.op, condition.value
        if op == "exists":
            return (raw is not None) is value
        if op in {"eq", "ne"}:
            result = self.equal(raw, value)
            return result if op == "eq" else not result
        if op in {"in", "not_in"}:
            result = any(self.equal(raw, candidate) for candidate in value)
            return result if op == "in" else not result
        if raw is None:
            return False
        if op == "regex":
            if value not in self.compiled:
                try:
                    self.compiled[value] = regex.compile(value)
                except regex.error as error:
                    raise FilterFailure(
                        "INVALID_REGEX", "A regular expression is invalid."
                    ) from error
            if self.regex_seconds >= REGEX_TOTAL_SECONDS:
                raise FilterFailure(
                    "REGEX_TIMEOUT", "Regular-expression filtering exceeded its time budget."
                )
            started = time.monotonic()
            try:
                return (
                    self.compiled[value].search(
                        str(raw),
                        timeout=min(
                            REGEX_TIMEOUT_SECONDS, REGEX_TOTAL_SECONDS - self.regex_seconds
                        ),
                    )
                    is not None
                )
            except TimeoutError as error:
                raise FilterFailure(
                    "REGEX_TIMEOUT", "A regular expression exceeded its time budget."
                ) from error
            finally:
                self.regex_seconds += time.monotonic() - started
        left, right = self.number(raw), self.number(value)
        return {"lt": left < right, "lte": left <= right, "gt": left > right, "gte": left >= right}[
            op
        ]

    def conjunction(self, row, conditions):
        # Evaluate every condition on this slide; never combine different slides'
        # evidence and never hide malformed expressions behind short-circuiting.
        return all([self.matches(row, condition) for condition in conditions])


def valid_patient(row):
    patient = row.get("patientId")
    return isinstance(patient, str) and bool(patient) and patient == patient.strip()


def development_selection_groups(rows):
    """Keep unresolved rows selectable without inventing patient identities.

    Temporary tuple keys cannot collide with supplied patient IDs. A selected
    unresolved row is rejected before any split generation; an unselected row
    does not become a prerequisite for a development cohort.
    """
    groups = defaultdict(list)
    for row in rows:
        key = (0, row["patientId"]) if valid_patient(row) else (1, row["slideId"])
        groups[key].append(row)
    return groups


def cohort_statistics(rows):
    verified = {
        row["patientId"]
        for row in rows
        if valid_patient(row) and row.get("patientIdSource") != "slide_fallback"
    }
    fallback = [row for row in rows if row.get("patientIdSource") == "slide_fallback"]
    return {
        "totalSlides": len(rows),
        "patientCount": len(verified),
        "fallbackSlideCount": len(fallback),
        "groupCount": len(verified)
        + len({row["patientId"] for row in fallback if valid_patient(row)}),
        "unlinkedSlideCount": sum(not valid_patient(row) for row in rows),
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


def fixed_assignments(groups, rules, evaluator, finding):
    """Shared rule evaluation for live feedback and authoritative protocol preview."""
    direct = {partition: [] for partition in PARTITIONS}
    expanded = {partition: [] for partition in PARTITIONS}
    assignments = {}
    for patient, group in sorted(groups.items()):
        matched = []
        for partition in PARTITIONS:
            conditions = getattr(rules, partition)
            if not conditions:
                continue
            # Evaluate all rows so invalid typed values cannot hide behind a match.
            rows = [row for row in group if evaluator.conjunction(row, conditions)]
            direct[partition].extend(rows)
            if rows:
                matched.append(partition)
                expanded[partition].extend(group)
        if len(matched) > 1:
            finding(
                "OVERLAPPING_PATIENT_RULES",
                "Fixed partition rules overlap after expansion to all eligible slides of a group.",
            )
        elif matched:
            assignments[patient] = matched[0]
    return assignments, direct, expanded


def _memberships(plans, groups, *, slide_unit):
    """Expand each plan's group roles to one frozen membership row per slide."""
    memberships = []
    for metadata, assignment in plans:
        plan = {key: value for key, value in metadata.items() if key != "excludedValidation"}
        for patient, partition in sorted(assignment.items()):
            for row in groups[patient]:
                memberships.append(
                    {
                        **plan,
                        "partition": partition,
                        "slideId": row["slideId"],
                        "patientId": row.get("patientId") if slide_unit else patient,
                        **(
                            {"patientIdSource": row["patientIdSource"]}
                            if "patientIdSource" in row
                            else {}
                        ),
                        "label": row["label"],
                    }
                )
    return sorted(
        memberships,
        key=lambda item: (
            item["seed"],
            item.get("planId", ""),
            -1 if item["fold"] is None else item["fold"],
            item["partition"],
            item["patientId"] or "",
            item["slideId"],
        ),
    )


class ProtocolService:
    def __init__(self, store: ScientificStore, filesystem: LocalFilesystem | None = None):
        self.store = store
        self.filesystem = filesystem or LocalFilesystem(())

    def _load(self, draft_id, expected_revision, allow_frozen):
        if type(expected_revision) is not int or expected_revision < 1:
            raise _failure("Supply a positive draft revision.", "INVALID_REVISION", 422)
        draft = self.store.get_draft(draft_id)
        replay = (
            allow_frozen
            and draft["status"] == "frozen"
            and draft["revision"] == expected_revision + 1
        )
        if draft["revision"] != expected_revision and not replay:
            raise _failure("The draft changed. Reload before previewing.", "REVISION_CONFLICT")
        if draft["status"] != "editable" and not replay:
            raise _failure("A frozen protocol draft cannot be edited.", "DRAFT_FROZEN")
        payload = draft["payload"]
        if (
            draft["kind"] != "experiment"
            or not isinstance(payload, dict)
            or set(payload) != {"type", "spec"}
            or payload["type"] != "analysis-protocol"
        ):
            raise _failure(
                "Select an analysis-protocol experiment draft.", "INVALID_PROTOCOL_DRAFT", 422
            )
        try:
            spec = ProtocolSpec.model_validate(payload["spec"])
        except ValidationError as error:
            errors = error.errors(include_input=False, include_url=False)
            details = [
                f"{'.'.join(str(part) for part in item['loc']) or 'protocol'}: "
                f"{item['msg'].removeprefix('Value error, ')}"
                for item in errors[:8]
            ]
            if len(errors) > 8:
                details.append(f"{len(errors) - 8} more fields need correction.")
            raise _failure(
                "The protocol specification is invalid: " + "; ".join(details),
                "INVALID_PROTOCOL_SPEC",
                422,
            ) from error
        if spec.split.version != 4:
            # SplitSpec still parses versions 1-3 so frozen records remain readable.
            raise _failure(
                "This protocol was created before development-only splits; frozen versions "
                "stay readable, but it can no longer be previewed or frozen. Design training "
                "in Targets & splits.",
                "LEGACY_PROTOCOL_SPLIT",
                422,
            )
        dataset, fields, records = self._load_dataset(spec.datasetId)
        if spec.sourceTargetSplitId:
            from histopilot.application.target_split_source import restrict_target_split_rows

            records = restrict_target_split_rows(
                self.store,
                spec.sourceTargetSplitId,
                spec.datasetId,
                spec.target,
                records,
                "train",
                split_unit=spec.splitUnit,
            )
        return spec, dataset, fields, records

    def _load_dataset(self, dataset_id):
        dataset = self.store.get_dataset(dataset_id)
        manifest = dataset["manifest"]
        if manifest.get("kind") != "dataset" or not isinstance(manifest.get("dictionary"), list):
            raise _failure(
                "Select a frozen imported dataset with a data dictionary.", "INVALID_DATASET"
            )
        try:
            records = json.loads(self.store.read_artifact(dataset_id, "records.json"))
        except (ValueError, UnicodeError) as error:
            raise _failure("The frozen dataset records are invalid.", "STORAGE_CORRUPT") from error
        if not isinstance(records, list) or len(records) > MAX_RECORDS:
            raise _failure(
                "This protocol supports at most 50,000 slide records.", "PROTOCOL_RECORD_LIMIT", 413
            )
        fields = {}
        for column in manifest["dictionary"]:
            if (
                not isinstance(column, dict)
                or not isinstance(column.get("key"), str)
                or not isinstance(column.get("sourceColumn"), str)
                or column["key"] in fields
            ):
                raise _failure("The frozen data dictionary is invalid.", "STORAGE_CORRUPT")
            fields[column["key"]] = column
        for row in records:
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("slideId"), str)
                or not row["slideId"].strip()
                or not isinstance(row.get("attributes"), dict)
                or any(
                    value is not None and not isinstance(value, str)
                    for value in row["attributes"].values()
                )
                or row.get("patientId") is not None
                and not isinstance(row["patientId"], str)
            ):
                raise _failure(
                    "The frozen slide records have an invalid schema.", "STORAGE_CORRUPT"
                )
        return dataset, fields, sorted(records, key=lambda row: row["slideId"])

    def explore(self, request: ProtocolExploreRequest) -> dict:
        """Evaluate complete frozen data without creating a draft or requiring labels."""
        _dataset, fields, rows = self._load_dataset(request.datasetId)
        findings = []

        def finding(code, message, severity="error"):
            value = {"code": code, "message": message, "severity": severity}
            if value not in findings:
                findings.append(value)

        split = {} if request.cohortOnly else request.split or {}
        mode = split.get("mode", request.splitMode)
        if not isinstance(mode, str) or mode not in {
            "rules",
            "kfold",
            "holdout",
            "imported",
            "monte_carlo",
            "leave_one_domain_out",
            "nested_kfold",
            "held_out",
        }:
            finding(
                "INVALID_STRATEGY_CONFIG", "Choose a valid split strategy to preview assignments."
            )
            mode = request.splitMode
        if split and split.get("version") != 4:
            # Split versions 1-3 are frozen history: read and evaluated, never designed again.
            finding(
                "INVALID_STRATEGY_CONFIG",
                "Live counts are available for development (version 4) splits only.",
            )
        result = {
            "datasetId": request.datasetId,
            "splitMode": mode,
            "dataset": cohort_statistics(rows),
            "cohort": None,
            "partitions": None,
            "unassigned": None,
            "target": None,
            "findings": findings,
        }
        evaluator = FilterEvaluator()

        def validate(conditions):
            for condition in iter_conditions(conditions):
                if condition.field not in fields and condition.field not in CANONICAL:
                    raise FilterFailure(
                        "UNKNOWN_FIELD",
                        f"The field '{condition.field}' is not in the frozen dataset dictionary.",
                    )
            evaluator.prepare(conditions)

        try:
            validate(request.eligibility)
            eligible = [row for row in rows if evaluator.conjunction(row, request.eligibility)]
        except FilterFailure as error:
            finding(error.code, str(error))
            return {**result, "valid": False}
        result["cohort"] = cohort_statistics(eligible)
        if request.cohortOnly:
            # Unresolved patient identities can be filtered out when assigning
            # training/testing groups later. They do not invalidate this count.
            return {**result, "valid": True}
        if request.targetField:
            if request.targetField not in fields and request.targetField not in CANONICAL:
                finding(
                    "UNKNOWN_FIELD", "The selected target is not in the frozen dataset dictionary."
                )
            else:
                counts = Counter(evaluator.field(row, request.targetField) for row in eligible)
                result["target"] = {
                    "field": request.targetField,
                    "values": [
                        {"value": value, "slides": count}
                        for value, count in sorted(
                            counts.items(),
                            key=lambda item: (-item[1], item[0] is None, item[0] or ""),
                        )[:20]
                    ],
                    "distinctCount": len(counts),
                }
        development = split.get("version") == 4
        if development:
            groups = development_selection_groups(eligible)
        else:
            groups = defaultdict(list)
            for row in eligible:
                if valid_patient(row):
                    groups[row["patientId"]].append(row)
            self._identity_findings(eligible, groups, finding)
        if (not development and result["cohort"]["unlinkedSlideCount"]) or any(
            item["code"] == "INVALID_STRATEGY_CONFIG" for item in findings
        ):
            # Eligibility counts remain useful; unresolved identities cannot be
            # silently treated as acknowledged independent groups in partition counts.
            return {**result, "valid": False}
        if not development:
            # Without a development split, live feedback covers the cohort and target only.
            return {**result, "valid": not any(item["severity"] == "error" for item in findings)}
        try:
            pools = PoolSpec.model_validate(split.get("pools", {}))
            validate([condition for role in PARTITIONS for condition in getattr(pools.rules, role)])
            if (
                pools.imported
                and pools.imported.partitionField not in fields
                and pools.imported.partitionField not in CANONICAL
            ):
                raise FilterFailure(
                    "UNKNOWN_FIELD", "The predefined pool column is not in the frozen dataset."
                )
            _assignments, direct, expanded, remaining = select_development_pools(
                groups, pools, evaluator, finding, fixed_assignments
            )
            selected_rows = expanded["train"] + expanded["val"]
            self._identity_findings(
                selected_rows,
                {key: rows for key, rows in groups.items() if key in _assignments},
                finding,
            )
        except ValidationError:
            finding(
                "INVALID_POOL_SETTINGS",
                "Complete the development source settings to see matching counts.",
            )
            return {**result, "valid": False, "selectionBasis": "pools"}
        except FilterFailure as error:
            finding(error.code, str(error))
            return {**result, "valid": False, "selectionBasis": "pools"}
        if any(
            item["code"]
            in {
                "OVERLAPPING_PATIENT_RULES",
                "IMPORTED_PATIENT_LEAKAGE",
                "MISSING_PATIENT_ID",
                "FALLBACK_PATIENT_ID_COLLISION",
            }
            for item in findings
        ):
            return {**result, "valid": False, "selectionBasis": "pools"}
        result["partitions"] = {
            role: {
                "selection": "remaining"
                if role == "train"
                and pools.source == "rules"
                and pools.trainSelection == "remaining"
                else "rules"
                if direct[role]
                else "none",
                "directMatches": cohort_statistics(
                    sorted(direct[role], key=lambda row: row["slideId"])
                ),
                "expanded": cohort_statistics(
                    sorted(expanded[role], key=lambda row: row["slideId"])
                ),
            }
            for role in PARTITIONS
        }
        result["unassigned"] = cohort_statistics(remaining)
        if result["target"]:
            counts = Counter(evaluator.field(row, request.targetField) for row in selected_rows)
            result["target"] = {
                "field": request.targetField,
                "values": [
                    {"value": value, "slides": count}
                    for value, count in sorted(
                        counts.items(),
                        key=lambda item: (-item[1], item[0] is None, item[0] or ""),
                    )[:20]
                ],
                "distinctCount": len(counts),
            }
        return {
            **result,
            "selectionBasis": "pools",
            "message": "Development plans use the selected training groups and early-stop validation. Unmatched rows stay outside this protocol.",
            "valid": not any(item["severity"] == "error" for item in findings),
        }

    @staticmethod
    def _identity_findings(rows, groups, finding):
        if any(not valid_patient(row) for row in rows):
            finding(
                "MISSING_PATIENT_ID",
                "Some slides have unresolved Patient_ID. Revise the dataset to map patients or explicitly confirm Slide_ID fallback.",
            )
        fallback = [row for row in rows if row.get("patientIdSource") == "slide_fallback"]
        if fallback:
            finding(
                "SLIDE_ID_FALLBACK_GROUPING",
                f"{len(fallback)} slides use acknowledged Slide_ID fallback groups. Patient independence cannot be verified for these slides.",
                "warning",
            )
        for group in groups.values():
            fallback_rows = [row for row in group if row.get("patientIdSource") == "slide_fallback"]
            if fallback_rows and (
                len(group) != 1 or fallback_rows[0]["patientId"] != fallback_rows[0]["slideId"]
            ):
                finding(
                    "FALLBACK_PATIENT_ID_COLLISION",
                    "A Slide_ID fallback collides with another grouping identity or differs from its slide ID. Revise the dataset before assigning partitions.",
                )

    def preview(self, draft_id: str, expected_revision: int) -> dict:
        return self._preview(draft_id, expected_revision, allow_frozen=False)

    @staticmethod
    def _field_findings(spec, dataset, fields, conditions, finding):
        """Reject unknown fields and identifier, target or split leakage before reading rows."""
        split = spec.split
        imported = split.pools.imported
        assignment_fields = (
            {imported.partitionField, imported.foldField} - {None} if imported else set()
        )
        split_fields = set(assignment_fields)
        if split.domainField:
            split_fields.add(split.domainField)
        selected_fields = {
            spec.target.field,
            *spec.predictors,
            *(condition.field for condition in iter_conditions(conditions)),
            *split_fields,
        }
        for field in sorted(selected_fields):
            if field not in fields and field not in CANONICAL:
                finding(
                    "UNKNOWN_FIELD", f"The field '{field}' is not in the frozen dataset dictionary."
                )

        def source(field):
            return fields.get(field, {}).get("sourceColumn", field)

        provenance_mapping = dataset["manifest"].get("provenance", {}).get("mapping", {})
        source_identifiers = {
            name_key(provenance_mapping[name])
            for name in (
                "slideIdColumn",
                "patientIdColumn",
                "patientSourceSlideIdColumn",
                "patientSourcePatientIdColumn",
            )
            if isinstance(provenance_mapping.get(name), str)
        }
        target_source = source(spec.target.field)
        if (
            spec.target.field in CANONICAL
            or forbidden_name(spec.target.field, target=True)
            or forbidden_name(target_source, target=True)
            or name_key(target_source) in source_identifiers
        ):
            finding(
                "IDENTIFIER_TARGET", "Identifiers and partition fields cannot serve as the target."
            )
        if split.domainField:
            domain_source = source(split.domainField)
            if (
                split.domainField in CANONICAL
                or forbidden_name(split.domainField)
                or forbidden_name(domain_source)
            ):
                finding(
                    "INVALID_DOMAIN_FIELD",
                    "Choose a site or cohort attribute, rather than an identifier or partition column.",
                )
            if spec.target.field == split.domainField or name_key(target_source) == name_key(
                domain_source
            ):
                finding(
                    "DOMAIN_TARGET_LEAKAGE",
                    "The held-out site or cohort column cannot also be the prediction target.",
                )
            if name_key(domain_source) in source_identifiers:
                finding(
                    "INVALID_DOMAIN_FIELD",
                    "The site or cohort column cannot be an identity mapping source.",
                )
        split_sources = {name_key(source(field)) for field in split_fields}
        if spec.target.field in assignment_fields or name_key(target_source) in {
            name_key(source(field)) for field in assignment_fields
        }:
            finding(
                "SPLIT_TARGET_LEAKAGE",
                "An imported partition or fold column cannot also be the prediction target.",
            )
        for field in spec.predictors:
            predictor_source = source(field)
            if field == spec.target.field or name_key(predictor_source) == name_key(target_source):
                finding(
                    "TARGET_PREDICTOR_LEAKAGE",
                    f"Predictor '{field}' is the target or a target alias.",
                )
            elif (
                field in CANONICAL
                or field in split_fields
                or name_key(predictor_source) in split_sources
                or name_key(predictor_source) in source_identifiers
                or forbidden_name(field)
                or forbidden_name(predictor_source)
            ):
                finding(
                    "FORBIDDEN_PREDICTOR",
                    f"Predictor '{field}' is an identifier, label, or split field.",
                )

    @staticmethod
    def _labelled(target, eligible, evaluator, finding):
        """Map raw target values; missing or unmapped labels block unless explicitly excluded."""
        included, exclusions = [], Counter()
        blocked = False
        for row in eligible:
            raw = evaluator.field(row, target.field)
            if raw is None:
                exclusions["missingLabel"] += 1
                if target.missing == "block":
                    blocked = True
                    finding(
                        "MISSING_LABEL",
                        "Eligible slides have missing target labels; resolve or explicitly exclude them.",
                    )
                continue
            if raw not in target.labels:
                exclusions["unmappedLabel"] += 1
                if target.unmapped == "block":
                    blocked = True
                    finding(
                        "UNMAPPED_LABEL",
                        "Eligible slides have unmapped target labels; map or explicitly exclude them.",
                    )
                continue
            included.append({**row, "label": target.labels[raw]})
        if exclusions and not blocked:
            finding(
                "EXPLICIT_LABEL_EXCLUSIONS",
                f"Explicit label policies exclude {sum(exclusions.values())} eligible slides.",
                "warning",
            )
        return included, exclusions

    @staticmethod
    def _label_findings(spec, groups, included, evaluator, finding, *, patient_folds):
        """Check mixed-label groups, target-equivalent predictors and class minimums."""
        mixed_groups = sum(len({row["label"] for row in group}) > 1 for group in groups.values())
        mixed_slide_target = spec.target.unit == "slide" and mixed_groups > 0
        if mixed_slide_target:
            stratification = (
                "Stratification uses each patient's observed label combination. "
                if spec.split.stratify
                else "Label stratification is disabled. "
            )
            finding(
                "MIXED_SLIDE_LABEL_PATIENT_GROUPS",
                f"{mixed_groups} patient groups contain different slide labels. Slides keep their own labels and patients stay together. "
                + stratification
                + "Class counts count a patient once in each represented class. Patient-level grade metrics are unavailable for conflicting labels.",
                "warning",
            )
        elif mixed_groups:
            finding(
                "MIXED_PATIENT_LABELS",
                "A patient has conflicting mapped target labels. No majority label is selected; mixed-label slide-target stratification requires a version-4 slide target.",
            )
        for field in spec.predictors:
            pairs = [(evaluator.field(row, field), row["label"]) for row in included]
            if pairs and all(value is not None for value, _ in pairs):
                forward, backward = defaultdict(set), defaultdict(set)
                for value, label in pairs:
                    forward[value].add(label)
                    backward[label].add(value)
                if len(forward) == len(spec.target.classes) == len(backward) and all(
                    len(values) == 1 for values in (*forward.values(), *backward.values())
                ):
                    finding(
                        "TARGET_EQUIVALENT_PREDICTOR",
                        f"Predictor '{field}' is a one-to-one encoding of the mapped target in this cohort.",
                    )
        patient_counts = (
            group_class_counts(groups)
            if mixed_slide_target
            else Counter(
                group[0]["label"]
                for group in groups.values()
                if len({row["label"] for row in group}) == 1
            )
        )
        for label in spec.target.classes:
            if patient_counts[label] < spec.constraints.minPatientsPerClass:
                finding(
                    "INSUFFICIENT_CLASS_PATIENTS",
                    f"Class '{label}' has fewer than {spec.constraints.minPatientsPerClass} independent {'patients' if patient_folds else 'slides'}.",
                )
        return mixed_slide_target, patient_counts

    def _preview(self, draft_id, expected_revision, *, allow_frozen):
        spec, dataset, fields, rows = self._load(draft_id, expected_revision, allow_frozen)
        slide_unit = spec.splitUnit == "slide"
        # Folds and early-stop validation use patient groups unless a slide design
        # assigns slides independently. Slide targets keep per-slide labels either way.
        patient_folds = not slide_unit or spec.split.groupByPatient
        findings = []

        def finding(code, message, severity="error"):
            # Shared CV code retains its historical internal assessment key.
            # Translate role wording without changing user field/class names.
            message = message.replace(
                ": reported test requests", ": development assessment requests"
            ).replace(": test has fewer than", ": development assessment has fewer than")
            if not any(item["code"] == code and item["message"] == message for item in findings):
                findings.append({"severity": severity, "code": code, "message": message})

        def blocked():
            return any(item["severity"] == "error" for item in findings)

        pools = spec.split.pools
        conditions = [
            *spec.eligibility,
            *(condition for role in PARTITIONS for condition in getattr(pools.rules, role)),
        ]
        self._field_findings(spec, dataset, fields, conditions, finding)
        if len({row["slideId"] for row in rows}) != len(rows):
            finding("DUPLICATE_SLIDE_ID", "The frozen dataset contains duplicate slide identities.")
        evaluator = FilterEvaluator()
        eligible = []
        if not any(item["code"] == "UNKNOWN_FIELD" for item in findings):
            try:
                evaluator.prepare(conditions)
                eligible = [row for row in rows if evaluator.conjunction(row, spec.eligibility)]
            except FilterFailure as error:
                finding(error.code, str(error))
        selected_pools = {}
        if not blocked():
            # Select development sources before interpreting their labels. A
            # combined metadata file can contain unrelated, unlabeled rows.
            source_groups = (
                development_selection_groups(eligible)
                if patient_folds
                else {(0, row["slideId"]): [row] for row in eligible}
            )
            try:
                selected, _direct, expanded, _remaining = select_development_pools(
                    source_groups, pools, evaluator, finding, fixed_assignments
                )
                eligible = expanded["train"] + expanded["val"]
                if patient_folds:
                    self._identity_findings(
                        eligible,
                        {key: rows for key, rows in source_groups.items() if key in selected},
                        finding,
                    )
                selected_pools = {key[1]: role for key, role in selected.items() if key[0] == 0}
            except FilterFailure as error:
                finding(error.code, str(error))
        included, exclusion_counts = self._labelled(spec.target, eligible, evaluator, finding)
        if not included:
            finding("EMPTY_COHORT", "No slides remain after eligibility and target mapping.")
        groups = defaultdict(list)
        for row in included:
            if not patient_folds:
                groups[row["slideId"]].append(row)
            elif valid_patient(row):
                groups[row["patientId"]].append(row)
        if patient_folds:
            self._identity_findings(included, groups, finding)
        mixed_slide_target, patient_counts = self._label_findings(
            spec, groups, included, evaluator, finding, patient_folds=patient_folds
        )
        pool_assignments = {}
        if not blocked():
            pool_assignments = {
                patient: role for patient, role in selected_pools.items() if patient in groups
            }
            for role in ("train", "val"):
                if (
                    role == "train" or pools.validationSource == "fixed"
                ) and role not in pool_assignments.values():
                    finding(
                        f"{role.upper()}_POOL_EMPTY",
                        f"The selected {role} pool has no groups after target mapping.",
                    )
        memberships, partitions, plans = [], [], []
        training_groups = {
            patient: rows
            for patient, rows in groups.items()
            if pool_assignments.get(patient) == "train"
        }
        if not blocked():
            plans, training_groups = development_assignments(
                spec,
                groups,
                pool_assignments,
                evaluator,
                finding,
                MAX_MEMBERSHIPS,
                MAX_PROTOCOL_BYTES,
            )
            memberships = _memberships(plans, groups, slide_unit=slide_unit)
            partitions = [
                check_modern_plan(spec, groups, metadata, assignment, finding)
                for metadata, assignment in plans
            ]
        if spec.sourceTargetSplitId and not blocked():
            # _load already restricted rows to the exact frozen training set.
            # CV roles may vary across plans; selection or eligibility must not
            # silently discard part of that population while retaining its source.
            if {row["slideId"] for row in memberships} != {row["slideId"] for row in rows}:
                finding(
                    "TARGET_SPLIT_TRAINING_MEMBERSHIP_CHANGED",
                    "Training design must preserve every frozen training slide. "
                    "Change the training/testing selection in Targets & Splits instead.",
                )
        findings.sort(key=lambda item: (item["severity"], item["code"], item["message"]))
        algorithm = ALGORITHM_V4_MIXED if mixed_slide_target else ALGORITHM_V4
        cohort_stats = cohort_statistics(included)
        class_counts = Counter(row["label"] for row in included)
        summary = {
            "datasetId": spec.datasetId,
            "totalSlides": len(rows),
            "eligibleSlides": len(eligible),
            "includedSlides": len(included),
            "includedPatients": 0 if slide_unit else cohort_stats["patientCount"],
            "includedGroups": len(groups),
            "fallbackSlideCount": cohort_stats["fallbackSlideCount"],
            "unlinkedSlideCount": cohort_stats["unlinkedSlideCount"],
            "excludedSlides": len(rows) - len(included),
            "labelExclusions": dict(exclusion_counts),
            "classCounts": {label: class_counts[label] for label in spec.target.classes},
            "patientClassCounts": {}
            if slide_unit
            else {label: patient_counts[label] for label in spec.target.classes},
            "grouping": "slide_labels_patient_folds"
            if slide_unit and patient_folds
            else "slide"
            if slide_unit
            else "patient_with_slide_fallback"
            if cohort_stats["fallbackSlideCount"]
            else "patient",
            "targetUnit": spec.target.unit,
            "algorithm": algorithm,
            # Development splits have no fixed partition rules; frozen summaries keep the key.
            "fixedPatients": dict.fromkeys(PARTITIONS, 0),
            **modern_summary(spec, training_groups, plans),
            "poolCounts": pool_counts(
                groups, pool_assignments, spec.target.classes, split_unit=spec.splitUnit
            ),
            **development_summary(spec),
        }
        result = {
            "spec": _serialized_spec(spec),
            "summary": summary,
            "findings": findings,
            "canFreeze": not blocked(),
            "partitions": partitions,
            "memberships": memberships,
            "executionEnabled": False,
        }
        if len(_json(result)) > MAX_PROTOCOL_BYTES:
            finding(
                "PROTOCOL_DOCUMENT_LIMIT",
                "The protocol exceeds the 15 MiB document limit; reduce seeds, folds, or cohort size.",
            )
            result["canFreeze"] = False
            result["memberships"] = []
        digest_input = {
            **result,
            "algorithm": algorithm,
            "datasetContentHash": dataset["contentHash"],
        }
        return {**result, "previewHash": hashlib.sha256(_json(digest_input)).hexdigest()}

    def freeze(
        self,
        draft_id: str,
        expected_revision: int,
        preview_hash: str,
        operation_id: str,
        *,
        version_label: dict | None = None,
    ) -> dict:
        prior = self.store.configuration_publication(operation_id)
        if prior is not None:
            manifest = prior["manifest"]
            if manifest.get("kind") != "protocol" or manifest.get("previewHash") != preview_hash:
                raise _failure(
                    "This operation ID belongs to a different protocol freeze request.",
                    "OPERATION_CONFLICT",
                )
            # A completed request replays its immutable result. Live pack/source
            # freshness belongs to experiment input checks and cannot rewrite a frozen protocol.
            # Publication validates the original draft/revision/tag/note intent.
            return self.store.publish_configuration(
                draft_id,
                expected_revision=expected_revision,
                manifest=manifest,
                operation_id=operation_id,
                version_label=version_label,
            )
        preview = self._preview(draft_id, expected_revision, allow_frozen=True)
        if preview["previewHash"] != preview_hash:
            raise _failure(
                "The protocol preview changed. Review a fresh preview before freezing.",
                "STALE_PREVIEW",
            )
        if not preview["canFreeze"]:
            raise _failure(
                "Resolve all blocking protocol preflight findings before freezing.",
                "PROTOCOL_PREFLIGHT_BLOCKED",
            )
        manifest = {
            "kind": "protocol",
            "datasetId": preview["spec"]["datasetId"],
            "algorithm": preview["summary"]["algorithm"],
            "spec": preview["spec"],
            "previewHash": preview["previewHash"],
            "summary": preview["summary"],
            "partitions": preview["partitions"],
            "memberships": preview["memberships"],
            "findings": preview["findings"],
            "executionEnabled": False,
        }
        return self.store.publish_configuration(
            draft_id,
            expected_revision=expected_revision,
            manifest=manifest,
            operation_id=operation_id,
            version_label=version_label,
        )
