"""Explicit targets, patient-grouped splits and immutable protocol publication.

The algorithm sorts stable grouping identities and uses SHA-256 ordering. A dataset
may explicitly record acknowledged slide-ID fallback groups. Labels are never
dropped implicitly and a majority label is never selected.
"""

import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation

import regex
from pydantic import ValidationError

from histopilot.application.development_splits import (
    ALGORITHM_V4,
    development_assignments,
    development_summary,
    select_development_pools,
)
from histopilot.application.explicit_pools import (
    ALGORITHM_V3,
    explicit_assignments,
    pool_counts,
    select_pools,
)
from histopilot.application.feature_packs import FeaturePackService
from histopilot.application.modern_splits import (
    ALGORITHM_V2,
    check_modern_plan,
    modern_assignments,
    modern_summary,
)
from histopilot.schemas.protocols import (
    Condition,
    FixedRules,
    PoolSpec,
    ProtocolExploreRequest,
    ProtocolSpec,
)
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore

ALGORITHM = "histopilot-patient-stratification-v1"
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
    if spec.featurePackId is None:
        value.pop("featurePackId", None)
    if spec.split.version == 1:
        # Preserve existing preview hashes and frozen retry behavior exactly.
        value["split"] = {
            key: value["split"][key]
            for key in ("mode", "folds", "seeds", "ratios", "rules", "imported")
        }
    elif spec.split.version == 2:
        value["split"].pop("pools", None)
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


def _key(value):
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _forbidden_name(value, *, target=False):
    name = _key(value)
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
        for condition in conditions:
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


def _valid_patient(row):
    patient = row.get("patientId")
    return isinstance(patient, str) and bool(patient) and patient == patient.strip()


def _development_selection_groups(rows):
    """Keep unresolved rows selectable without inventing patient identities.

    Temporary tuple keys cannot collide with supplied patient IDs. A selected
    unresolved row is rejected before any split generation; an unselected row
    does not become a prerequisite for a development cohort.
    """
    groups = defaultdict(list)
    for row in rows:
        key = (0, row["patientId"]) if _valid_patient(row) else (1, row["slideId"])
        groups[key].append(row)
    return groups


def _cohort_stats(rows):
    verified = {
        row["patientId"]
        for row in rows
        if _valid_patient(row) and row.get("patientIdSource") != "slide_fallback"
    }
    fallback = [row for row in rows if row.get("patientIdSource") == "slide_fallback"]
    return {
        "totalSlides": len(rows),
        "patientCount": len(verified),
        "fallbackSlideCount": len(fallback),
        "groupCount": len(verified)
        + len({row["patientId"] for row in fallback if _valid_patient(row)}),
        "unlinkedSlideCount": sum(not _valid_patient(row) for row in rows),
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


def _fixed_assignments(groups, rules, evaluator, finding):
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
        dataset, fields, records = self._load_dataset(spec.datasetId)
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

        split = request.split or {}
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
        if "version" in split and (
            type(split["version"]) is not int or split["version"] not in (1, 2, 3, 4)
        ):
            finding("INVALID_STRATEGY_CONFIG", "The split version must be 1, 2, 3 or 4.")
        if "heldOutSource" in split and (
            not isinstance(split["heldOutSource"], str)
            or split["heldOutSource"] not in {"fractions", "rules", "imported"}
        ):
            finding("INVALID_STRATEGY_CONFIG", "Choose fractions, rules or predefined partitions.")
        modern = split.get("version") in (2, 3, 4) or mode in {
            "monte_carlo",
            "leave_one_domain_out",
            "nested_kfold",
            "held_out",
        }
        uses_rules = mode == "rules" or (
            modern and mode == "held_out" and split.get("heldOutSource") == "rules"
        )
        result = {
            "datasetId": request.datasetId,
            "splitMode": mode,
            "dataset": _cohort_stats(rows),
            "cohort": None,
            "partitions": None,
            "unassigned": None,
            "target": None,
            "findings": findings,
        }
        evaluator = FilterEvaluator()

        def validate(conditions):
            for condition in conditions:
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
        result["cohort"] = _cohort_stats(eligible)
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
        groups = defaultdict(list)
        if development:
            groups = _development_selection_groups(eligible)
        else:
            for row in eligible:
                if _valid_patient(row):
                    groups[row["patientId"]].append(row)
            self._identity_findings(eligible, groups, finding)
        if (not development and result["cohort"]["unlinkedSlideCount"]) or any(
            item["code"] == "INVALID_STRATEGY_CONFIG" for item in findings
        ):
            # Eligibility counts remain useful; unresolved identities cannot be
            # silently treated as acknowledged independent groups in partition counts.
            return {**result, "valid": False}
        if split.get("version") in (3, 4):
            development = split.get("version") == 4
            try:
                pools = PoolSpec.model_validate(split.get("pools", {}))
                validate(
                    [condition for role in PARTITIONS for condition in getattr(pools.rules, role)]
                )
                if (
                    pools.imported
                    and pools.imported.partitionField not in fields
                    and pools.imported.partitionField not in CANONICAL
                ):
                    raise FilterFailure(
                        "UNKNOWN_FIELD", "The predefined pool column is not in the frozen dataset."
                    )
                selector = select_development_pools if development else select_pools
                _assignments, direct, expanded, remaining = selector(
                    groups, pools, evaluator, finding, _fixed_assignments
                )
                if development:
                    selected_rows = expanded["train"] + expanded["val"]
                    self._identity_findings(
                        selected_rows,
                        {key: rows for key, rows in groups.items() if key in _assignments},
                        finding,
                    )
            except ValidationError:
                finding(
                    "INVALID_POOL_SETTINGS",
                    "Complete the development source settings to see matching counts."
                    if development
                    else "Complete the training and test pool settings to see matching counts.",
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
                    "directMatches": _cohort_stats(
                        sorted(direct[role], key=lambda row: row["slideId"])
                    ),
                    "expanded": _cohort_stats(
                        sorted(expanded[role], key=lambda row: row["slideId"])
                    ),
                }
                for role in PARTITIONS
            }
            result["unassigned"] = _cohort_stats(remaining)
            if development and result["target"]:
                selected_rows = expanded["train"] + expanded["val"]
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
                "message": "Development plans use the selected training groups and early-stop validation. Unmatched rows stay outside this protocol."
                if development
                else "Cross-validation uses the selected training pool. The external test pool is reserved for final evaluation. Validation comes from training unless a fixed validation pool is selected.",
                "valid": not any(item["severity"] == "error" for item in findings),
            }
        if modern and not uses_rules:
            return {
                **result,
                "message": "Preview the complete strategy to see training, early-stop validation and test assignments.",
                "valid": not any(item["severity"] == "error" for item in findings),
            }
        try:
            rules = FixedRules.model_validate(split["rules"]) if "rules" in split else request.rules
            validate([condition for role in PARTITIONS for condition in getattr(rules, role)])
            assignments, direct, expanded = _fixed_assignments(groups, rules, evaluator, finding)
        except ValidationError:
            finding("INVALID_FIXED_RULES", "Complete the fixed rules to see set counts.")
            return {**result, "valid": False}
        except FilterFailure as error:
            finding(error.code, str(error))
            return {**result, "valid": False}
        if any(item["severity"] == "error" for item in findings):
            return {**result, "valid": False}
        for role in PARTITIONS:
            if getattr(rules, role) and not expanded[role]:
                finding(
                    "EMPTY_FIXED_RULE",
                    f"The fixed {role} rule does not select any eligible groups.",
                )
        remaining = [row for row in eligible if row["patientId"] not in assignments]
        if uses_rules and not rules.train:
            expanded["train"] = remaining
            remaining = []
        elif uses_rules and remaining:
            finding(
                "UNASSIGNED_RULE_GROUPS",
                "Some eligible groups match no partition. Change the rules or leave training rules empty to use the remaining cohort.",
            )
        result["partitions"] = {
            role: {
                "selection": "rules"
                if getattr(rules, role)
                else "remaining"
                if role == "train" and uses_rules
                else "none",
                "directMatches": _cohort_stats(
                    sorted(direct[role], key=lambda row: row["slideId"])
                ),
                "expanded": _cohort_stats(sorted(expanded[role], key=lambda row: row["slideId"])),
            }
            for role in PARTITIONS
        }
        result["unassigned"] = _cohort_stats(remaining)
        if modern and not rules.val:
            result["message"] = (
                "Training counts show the development pool. Preview reserves the selected percentage for early stopping."
            )
        return {**result, "valid": not any(item["severity"] == "error" for item in findings)}

    @staticmethod
    def _identity_findings(rows, groups, finding):
        if any(not _valid_patient(row) for row in rows):
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

    def _preview(self, draft_id, expected_revision, *, allow_frozen):
        spec, dataset, fields, rows = self._load(draft_id, expected_revision, allow_frozen)
        modern = spec.split.version >= 2
        explicit = spec.split.version >= 3
        development = spec.split.version == 4
        algorithm = (
            ALGORITHM_V4
            if development
            else ALGORITHM_V3
            if explicit
            else ALGORITHM_V2
            if modern
            else ALGORITHM
        )
        findings = []

        def finding(code, message, severity="error"):
            if development:
                # Shared CV code retains its historical internal assessment key.
                # Translate role wording without changing user field/class names.
                message = message.replace(
                    ": reported test requests", ": development assessment requests"
                ).replace(": test has fewer than", ": development assessment has fewer than")
            if not any(item["code"] == code and item["message"] == message for item in findings):
                findings.append({"severity": severity, "code": code, "message": message})

        conditions = [
            *spec.eligibility,
            *spec.split.rules.train,
            *spec.split.rules.val,
            *spec.split.rules.test,
        ]
        if explicit:
            conditions.extend(
                condition
                for role in PARTITIONS
                for condition in getattr(spec.split.pools.rules, role)
            )
        selected_fields = [
            spec.target.field,
            *spec.predictors,
            *(condition.field for condition in conditions),
        ]
        imported = spec.split.pools.imported if explicit else spec.split.imported
        if imported:
            selected_fields.extend(
                field for field in (imported.partitionField, imported.foldField) if field
            )
        if modern and spec.split.domainField:
            selected_fields.append(spec.split.domainField)
        for field in sorted(set(selected_fields)):
            if field not in fields and field not in CANONICAL:
                finding(
                    "UNKNOWN_FIELD", f"The field '{field}' is not in the frozen dataset dictionary."
                )
        if spec.target.field in CANONICAL or _forbidden_name(spec.target.field, target=True):
            finding(
                "IDENTIFIER_TARGET", "Identifiers and partition fields cannot serve as the target."
            )
        target_source = fields.get(spec.target.field, {}).get("sourceColumn", spec.target.field)
        if _forbidden_name(target_source, target=True):
            finding(
                "IDENTIFIER_TARGET", "Identifiers and partition fields cannot serve as the target."
            )
        assignment_fields = (
            {imported.partitionField, imported.foldField} - {None} if imported else set()
        )
        split_fields = set(assignment_fields)
        if modern and spec.split.domainField:
            split_fields.add(spec.split.domainField)
            domain_source = fields.get(spec.split.domainField, {}).get(
                "sourceColumn", spec.split.domainField
            )
            if (
                spec.split.domainField in CANONICAL
                or _forbidden_name(spec.split.domainField)
                or _forbidden_name(domain_source)
            ):
                finding(
                    "INVALID_DOMAIN_FIELD",
                    "Choose a site or cohort attribute, rather than an identifier or partition column.",
                )
            if spec.target.field == spec.split.domainField or _key(target_source) == _key(
                domain_source
            ):
                finding(
                    "DOMAIN_TARGET_LEAKAGE",
                    "The held-out site or cohort column cannot also be the prediction target.",
                )
        provenance_mapping = dataset["manifest"].get("provenance", {}).get("mapping", {})
        source_identifiers = {
            _key(provenance_mapping[name])
            for name in (
                "slideIdColumn",
                "patientIdColumn",
                "patientSourceSlideIdColumn",
                "patientSourcePatientIdColumn",
            )
            if isinstance(provenance_mapping.get(name), str)
        }
        if _key(target_source) in source_identifiers:
            finding(
                "IDENTIFIER_TARGET", "Identifiers and partition fields cannot serve as the target."
            )
        if modern and spec.split.domainField and _key(domain_source) in source_identifiers:
            finding(
                "INVALID_DOMAIN_FIELD",
                "The site or cohort column cannot be an identity mapping source.",
            )
        split_sources = {
            _key(fields.get(field, {}).get("sourceColumn", field)) for field in split_fields
        }
        assignment_sources = {
            _key(fields.get(field, {}).get("sourceColumn", field)) for field in assignment_fields
        }
        if spec.target.field in assignment_fields or _key(target_source) in assignment_sources:
            finding(
                "SPLIT_TARGET_LEAKAGE",
                "An imported partition or fold column cannot also be the prediction target.",
            )
        for field in spec.predictors:
            source = fields.get(field, {}).get("sourceColumn", field)
            if field == spec.target.field or _key(source) == _key(target_source):
                finding(
                    "TARGET_PREDICTOR_LEAKAGE",
                    f"Predictor '{field}' is the target or a target alias.",
                )
            elif (
                field in CANONICAL
                or field in split_fields
                or _key(source) in split_sources
                or _key(source) in source_identifiers
                or _forbidden_name(field)
                or _forbidden_name(source)
            ):
                finding(
                    "FORBIDDEN_PREDICTOR",
                    f"Predictor '{field}' is an identifier, label, or split field.",
                )
        if (
            not modern
            and spec.split.mode != "holdout"
            and spec.split.ratios.model_dump()
            != {
                "train": 0.8,
                "val": 0.2,
                "test": 0,
            }
        ):
            finding(
                "UNUSED_HOLDOUT_RATIOS",
                "Holdout ratios apply only to generated holdout mode; remove them or switch split mode.",
            )
        if len({row["slideId"] for row in rows}) != len(rows):
            finding("DUPLICATE_SLIDE_ID", "The frozen dataset contains duplicate slide identities.")
        evaluator = FilterEvaluator()
        eligible, included = [], []
        exclusion_counts = Counter()
        if not any(item["code"] == "UNKNOWN_FIELD" for item in findings):
            try:
                evaluator.prepare(conditions)
                eligible = [row for row in rows if evaluator.conjunction(row, spec.eligibility)]
            except FilterFailure as error:
                finding(error.code, str(error))
        development_pool_assignments = {}
        if development and not any(item["severity"] == "error" for item in findings):
            # Select development sources before interpreting their labels. A
            # combined metadata file can contain unrelated, unlabeled rows.
            source_groups = _development_selection_groups(eligible)
            try:
                selected, _direct, _expanded, _remaining = select_development_pools(
                    source_groups, spec.split.pools, evaluator, finding, _fixed_assignments
                )
                eligible = _expanded["train"] + _expanded["val"]
                self._identity_findings(
                    eligible,
                    {key: rows for key, rows in source_groups.items() if key in selected},
                    finding,
                )
                development_pool_assignments = {
                    key[1]: role for key, role in selected.items() if key[0] == 0
                }
            except FilterFailure as error:
                finding(error.code, str(error))
        for row in eligible:
            raw = evaluator.field(row, spec.target.field)
            if raw is None:
                exclusion_counts["missingLabel"] += 1
                if spec.target.missing == "block":
                    finding(
                        "MISSING_LABEL",
                        "Eligible slides have missing target labels; resolve or explicitly exclude them.",
                    )
                continue
            if raw not in spec.target.labels:
                exclusion_counts["unmappedLabel"] += 1
                if spec.target.unmapped == "block":
                    finding(
                        "UNMAPPED_LABEL",
                        "Eligible slides have unmapped target labels; map or explicitly exclude them.",
                    )
                continue
            included.append({**row, "label": spec.target.labels[raw]})
        if exclusion_counts and not any(
            item["code"] in {"MISSING_LABEL", "UNMAPPED_LABEL"} for item in findings
        ):
            finding(
                "EXPLICIT_LABEL_EXCLUSIONS",
                f"Explicit label policies exclude {sum(exclusion_counts.values())} eligible slides.",
                "warning",
            )
        if not included:
            finding("EMPTY_COHORT", "No slides remain after eligibility and target mapping.")
        if modern and not explicit and imported:
            # A predefined validation cohort remains an explicit scientific choice
            # when filtering or label policies exclude it. Do not replace it with
            # an automatic training subset simply because no validation rows remain.
            def is_imported_validation(row):
                return (
                    imported.partitionLabels.get(evaluator.field(row, imported.partitionField))
                    == "val"
                )

            if any(is_imported_validation(row) for row in rows) and not any(
                is_imported_validation(row) for row in included
            ):
                finding(
                    "EMPTY_IMPORTED_VALIDATION",
                    "All predefined validation slides were excluded by eligibility or target "
                    "mapping. Restore eligible validation groups or explicitly revise the "
                    "partition mapping; automatic validation will not replace this cohort.",
                )
        groups = defaultdict(list)
        for row in included:
            if _valid_patient(row):
                groups[row["patientId"]].append(row)
        self._identity_findings(included, groups, finding)
        for group in groups.values():
            if len({row["label"] for row in group}) != 1:
                code = (
                    "MIXED_PATIENT_LABELS"
                    if spec.target.unit == "patient"
                    else "MIXED_PATIENT_STRATIFICATION_UNSUPPORTED"
                )
                finding(
                    code,
                    "A patient has conflicting mapped target labels. No majority label is selected; mixed-label slide-target stratification is not supported yet.",
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
        class_counts = Counter(row["label"] for row in included)
        patient_counts = Counter(
            group[0]["label"]
            for group in groups.values()
            if len({row["label"] for row in group}) == 1
        )
        for label in spec.target.classes:
            if patient_counts[label] < spec.constraints.minPatientsPerClass:
                finding(
                    "INSUFFICIENT_CLASS_PATIENTS",
                    f"Class '{label}' has fewer than {spec.constraints.minPatientsPerClass} independent patients.",
                )
        fixed = {}
        try:
            fixed, _direct, _expanded = _fixed_assignments(
                groups, spec.split.rules, evaluator, finding
            )
        except FilterFailure as error:
            finding(error.code, str(error))
        for partition in PARTITIONS:
            if getattr(spec.split.rules, partition) and partition not in fixed.values():
                finding(
                    "EMPTY_FIXED_RULE",
                    f"The fixed {partition} rule does not select any included patients.",
                )
        pool_assignments = {}
        if explicit and not any(item["severity"] == "error" for item in findings):
            try:
                if development:
                    pool_assignments = {
                        patient: role
                        for patient, role in development_pool_assignments.items()
                        if patient in groups
                    }
                    for role in ("train", "val"):
                        if (
                            role == "train" or spec.split.pools.validationSource == "fixed"
                        ) and role not in pool_assignments.values():
                            finding(
                                f"{role.upper()}_POOL_EMPTY",
                                f"The selected {role} pool has no groups after target mapping.",
                            )
                else:
                    pool_assignments, _direct, _expanded, _remaining = select_pools(
                        groups, spec.split.pools, evaluator, finding, _fixed_assignments
                    )
            except FilterFailure as error:
                finding(error.code, str(error))
        feature_hash = None
        feature_pack = None
        if spec.featureSetId:
            feature = self.store.get_configuration(spec.featureSetId)
            feature_hash = feature["contentHash"]
            feature_manifest = feature["manifest"]
            if (
                feature_manifest.get("kind") != "feature"
                or feature_manifest.get("datasetId") != spec.datasetId
            ):
                finding(
                    "FEATURE_DATASET_MISMATCH",
                    "The selected feature set must belong to this frozen dataset.",
                )
            else:
                feature_slides = [item.get("slideId") for item in feature_manifest.get("files", [])]
                if len(feature_slides) != len(set(feature_slides)):
                    finding(
                        "DUPLICATE_FEATURE_ID",
                        "The selected feature set contains duplicate slide identities.",
                    )
                if set(row["slideId"] for row in included) - set(feature_slides):
                    finding(
                        "MISSING_FEATURE_COVERAGE",
                        "The selected feature set does not cover every included slide.",
                    )
                if spec.featurePackId:
                    try:
                        resolved = FeaturePackService(self.store, self.filesystem).resolve_artifact(
                            spec.featureSetId, spec.featurePackId
                        )
                        if resolved["current"]:
                            feature_pack = pack_binding_snapshot(resolved["artifact"])
                        else:
                            finding(
                                "FEATURE_PACK_UNAVAILABLE",
                                "The selected pack needs verification in PFM & features.",
                            )
                            findings.extend(
                                {**item, "severity": "error"} for item in resolved["findings"]
                            )
                    except StorageError as error:
                        finding(error.code, str(error))
                else:
                    finding(
                        "FEATURE_VALUES_UNVERIFIED",
                        "Feature attachment validation covers headers; execution still requires content and runtime preflight.",
                        "warning",
                    )
        else:
            finding(
                "FEATURE_SET_NOT_SELECTED",
                "A feature set can be selected later; this protocol does not authorize execution.",
                "warning",
            )
        memberships, partitions, plans = [], [], []
        training_groups = (
            {
                patient: rows
                for patient, rows in groups.items()
                if pool_assignments.get(patient) == "train"
            }
            if explicit
            else groups
        )
        if modern and not any(item["severity"] == "error" for item in findings):
            if explicit:
                assign = development_assignments if development else explicit_assignments
                plans, training_groups = assign(
                    spec,
                    groups,
                    pool_assignments,
                    evaluator,
                    finding,
                    MAX_MEMBERSHIPS,
                    MAX_PROTOCOL_BYTES,
                )
            else:
                plans = modern_assignments(
                    spec, groups, fixed, evaluator, finding, MAX_MEMBERSHIPS, MAX_PROTOCOL_BYTES
                )
            for metadata, assignment in plans:
                for patient, partition in sorted(assignment.items()):
                    for row in groups[patient]:
                        memberships.append(
                            {
                                **{
                                    key: value
                                    for key, value in metadata.items()
                                    if key != "excludedValidation"
                                },
                                "partition": partition,
                                "slideId": row["slideId"],
                                "patientId": patient,
                                **(
                                    {"patientIdSource": row["patientIdSource"]}
                                    if "patientIdSource" in row
                                    else {}
                                ),
                                "label": row["label"],
                            }
                        )
                partitions.append(check_modern_plan(spec, groups, metadata, assignment, finding))
        elif not modern and not any(item["severity"] == "error" for item in findings):
            iterations = len(spec.split.seeds) * (
                spec.split.folds
                if spec.split.mode == "kfold" or imported and imported.foldField
                else 1
            )
            estimated_bytes = 0
            for row in included:
                estimated_bytes += (
                    len(
                        _json(
                            {
                                "seed": 4294967295,
                                "fold": None,
                                "partition": "train",
                                "slideId": row["slideId"],
                                "patientId": row["patientId"],
                                "label": row["label"],
                            }
                        )
                    )
                    + 1
                )
                if estimated_bytes * iterations > MAX_PROTOCOL_BYTES:
                    break
            if (
                len(included)
                * len(spec.split.seeds)
                * (
                    spec.split.folds
                    if spec.split.mode == "kfold" or imported and imported.foldField
                    else 1
                )
                > MAX_MEMBERSHIPS
            ):
                finding(
                    "PROTOCOL_MEMBERSHIP_LIMIT",
                    "The requested seeds and folds exceed 500,000 explicit membership rows.",
                )
            elif estimated_bytes * iterations > MAX_PROTOCOL_BYTES:
                finding(
                    "PROTOCOL_DOCUMENT_LIMIT",
                    "The requested explicit memberships exceed the 15 MiB protocol limit; reduce seeds, folds, or cohort size.",
                )
            else:
                assignments = self._assign(spec, groups, fixed, evaluator, finding)
                for seed, fold, assignment in assignments:
                    for patient, partition in sorted(assignment.items()):
                        for row in groups[patient]:
                            memberships.append(
                                {
                                    "seed": seed,
                                    "fold": fold,
                                    "partition": partition,
                                    "slideId": row["slideId"],
                                    "patientId": patient,
                                    **(
                                        {"patientIdSource": row["patientIdSource"]}
                                        if "patientIdSource" in row
                                        else {}
                                    ),
                                    "label": row["label"],
                                }
                            )
                    partitions.append(
                        self._check_partition(spec, groups, seed, fold, assignment, finding)
                    )
        memberships.sort(
            key=lambda item: (
                item["seed"],
                item.get("planId", ""),
                -1 if item["fold"] is None else item["fold"],
                item["partition"],
                item["patientId"],
                item["slideId"],
            )
        )
        findings.sort(key=lambda item: (item["severity"], item["code"], item["message"]))
        cohort_stats = _cohort_stats(included)
        summary = {
            "datasetId": spec.datasetId,
            "totalSlides": len(rows),
            "eligibleSlides": len(eligible),
            "includedSlides": len(included),
            "includedPatients": cohort_stats["patientCount"],
            "includedGroups": len(groups),
            "fallbackSlideCount": cohort_stats["fallbackSlideCount"],
            "unlinkedSlideCount": cohort_stats["unlinkedSlideCount"],
            "excludedSlides": len(rows) - len(included),
            "labelExclusions": dict(exclusion_counts),
            "classCounts": {label: class_counts[label] for label in spec.target.classes},
            "patientClassCounts": {label: patient_counts[label] for label in spec.target.classes},
            "grouping": "patient_with_slide_fallback"
            if cohort_stats["fallbackSlideCount"]
            else "patient",
            "targetUnit": spec.target.unit,
            "algorithm": algorithm,
            "fixedPatients": {
                partition: sum(value == partition for value in fixed.values())
                for partition in PARTITIONS
            },
            **(modern_summary(spec, training_groups, plans) if modern else {}),
            **(
                {
                    "poolCounts": pool_counts(groups, pool_assignments, spec.target.classes),
                    "finalPlanCount": sum(
                        metadata["phase"] == "final" for metadata, _assignment in plans
                    ),
                    "validationSource": spec.split.pools.validationSource,
                    "evaluationScope": "Cross-validation within training; external test used only in final evaluation",
                    "validationFractionScope": "Percentage of each training subset after CV evaluation and tuning folds are removed; fixed validation overrides it",
                    "roleDescriptions": {
                        "train": "Model fitting",
                        "val": "Early stopping using the selected fixed validation pool or a percentage of training",
                        "test": "Within-training CV evaluation in training plans; reserved external test in final plans",
                        "tune": "Inner held-out data for model selection; outer and external test groups are excluded",
                    },
                }
                if explicit and not development
                else {}
            ),
            **(
                {
                    "poolCounts": pool_counts(groups, pool_assignments, spec.target.classes),
                    **development_summary(spec),
                }
                if development
                else {}
            ),
        }
        result = {
            "spec": _serialized_spec(spec),
            "summary": summary,
            "findings": findings,
            "canFreeze": not any(item["severity"] == "error" for item in findings),
            "partitions": partitions,
            "memberships": memberships,
            "executionEnabled": False,
        }
        if feature_pack is not None:
            result["featurePack"] = feature_pack
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
            "featureContentHash": feature_hash,
        }
        return {**result, "previewHash": hashlib.sha256(_json(digest_input)).hexdigest()}

    @staticmethod
    def _ordered(patients, seed, label):
        return sorted(
            patients,
            key=lambda patient: (
                hashlib.sha256(_json([ALGORITHM, seed, label, patient])).digest(),
                patient,
            ),
        )

    def _assign(self, spec, groups, fixed, evaluator, finding):
        if spec.split.mode == "imported":
            return self._imported(spec, groups, fixed, evaluator, finding)
        if spec.split.mode == "rules":
            assignment = dict(fixed)
            remaining = set(groups) - set(assignment)
            if spec.split.rules.train and remaining:
                finding(
                    "UNASSIGNED_RULE_GROUPS",
                    "Some eligible groups match no partition. Change the rules or leave training rules empty to use the remaining cohort.",
                )
                return []
            assignment.update({patient: "train" for patient in remaining})
            if len(spec.split.seeds) > 1:
                finding(
                    "RULE_ASSIGNMENTS_REUSED",
                    "Rule-based memberships are identical across the selected seeds.",
                    "warning",
                )
            return [(seed, 0, dict(assignment)) for seed in sorted(spec.split.seeds)]
        free_by_class = defaultdict(list)
        for patient, group in sorted(groups.items()):
            if patient not in fixed:
                free_by_class[group[0]["label"]].append(patient)
        results = []
        for seed in sorted(spec.split.seeds):
            if spec.split.mode == "kfold":
                if sum(map(len, free_by_class.values())) < spec.split.folds:
                    finding(
                        "INFEASIBLE_FOLDS",
                        "There are fewer unfixed patients than requested validation folds.",
                    )
                    continue
                fold_map, sizes = {}, [0] * spec.split.folds
                for label in spec.target.classes:
                    ordered = self._ordered(free_by_class[label], seed, label)
                    offset = min(range(spec.split.folds), key=lambda fold: (sizes[fold], fold))
                    for index, patient in enumerate(ordered):
                        fold = (offset + index) % spec.split.folds
                        fold_map[patient] = fold
                        sizes[fold] += 1
                for fold in range(spec.split.folds):
                    assignment = {
                        **fixed,
                        **{
                            patient: "val" if member_fold == fold else "train"
                            for patient, member_fold in fold_map.items()
                        },
                    }
                    results.append((seed, fold, assignment))
            else:
                assignment = dict(fixed)
                ratios = spec.split.ratios.model_dump()
                for label in spec.target.classes:
                    ordered = self._ordered(free_by_class[label], seed, label)
                    desired = {
                        partition: ratios[partition] * len(ordered) for partition in PARTITIONS
                    }
                    counts = {partition: math.floor(desired[partition]) for partition in PARTITIONS}
                    remainder = len(ordered) - sum(counts.values())
                    priorities = sorted(
                        PARTITIONS,
                        key=lambda partition: (
                            -(desired[partition] - counts[partition]),
                            PARTITIONS.index(partition),
                        ),
                    )
                    for partition in priorities[:remainder]:
                        counts[partition] += 1
                    needs = {
                        partition: max(
                            0,
                            spec.constraints.minPatientsPerClass
                            - sum(
                                fixed.get(patient) == partition and group[0]["label"] == label
                                for patient, group in groups.items()
                            ),
                        )
                        if ratios[partition] > 0
                        else 0
                        for partition in PARTITIONS
                    }
                    if sum(needs.values()) <= len(ordered):
                        for partition in PARTITIONS:
                            while counts[partition] < needs[partition]:
                                donors = [
                                    candidate
                                    for candidate in PARTITIONS
                                    if counts[candidate] > needs[candidate]
                                ]
                                donor = max(
                                    donors,
                                    key=lambda candidate: (
                                        counts[candidate] - desired[candidate],
                                        counts[candidate],
                                        -PARTITIONS.index(candidate),
                                    ),
                                )
                                counts[donor] -= 1
                                counts[partition] += 1
                    offset = 0
                    for partition in PARTITIONS:
                        for patient in ordered[offset : offset + counts[partition]]:
                            assignment[patient] = partition
                        offset += counts[partition]
                results.append((seed, None, assignment))
        return results

    @staticmethod
    def _imported(spec, groups, fixed, evaluator, finding):
        imported = spec.split.imported
        states = {}
        for patient, group in sorted(groups.items()):
            patient_states = set()
            for row in group:
                partition = None
                if imported.partitionField:
                    raw = evaluator.field(row, imported.partitionField)
                    if raw is None or raw not in imported.partitionLabels:
                        finding(
                            "INVALID_IMPORTED_PARTITION",
                            "Every included slide needs an explicitly mapped imported partition.",
                        )
                        continue
                    partition = imported.partitionLabels[raw]
                if imported.foldField:
                    raw_fold = evaluator.field(row, imported.foldField)
                    if raw_fold in imported.testFoldLabels:
                        if partition not in (None, "test"):
                            finding(
                                "IMPORTED_PARTITION_FOLD_CONFLICT",
                                "A held-out fold conflicts with its imported partition.",
                            )
                        state = ("fixed", "test")
                    elif partition in ("train", "val", "test"):
                        if raw_fold is not None:
                            finding(
                                "IMPORTED_PARTITION_FOLD_CONFLICT",
                                "Fixed imported partitions require an empty fold or an explicit test-fold label.",
                            )
                        state = ("fixed", partition)
                    elif raw_fold is None or raw_fold not in imported.foldLabels:
                        finding(
                            "INVALID_IMPORTED_FOLD",
                            "Every development slide needs an explicit valid fold mapping; held-out labels require an explicit mapping.",
                        )
                        continue
                    else:
                        state = ("fold", imported.foldLabels[raw_fold])
                elif partition == "trainval":
                    finding(
                        "INVALID_IMPORTED_PARTITION",
                        "A trainval partition requires an imported fold field.",
                    )
                    continue
                else:
                    state = ("fixed", partition)
                patient_states.add(state)
            if len(patient_states) > 1:
                finding(
                    "IMPORTED_PATIENT_LEAKAGE",
                    "An imported split assigns slides from the same patient to different folds or partitions.",
                )
            elif patient_states:
                state = next(iter(patient_states))
                if patient in fixed and state != ("fixed", fixed[patient]):
                    finding(
                        "IMPORTED_RULE_CONFLICT",
                        "Imported assignments conflict with a patient-expanded fixed rule.",
                    )
                states[patient] = state
        if imported.foldField:
            used = {value for kind, value in states.values() if kind == "fold"}
            if used != set(range(spec.split.folds)):
                finding(
                    "INCOMPLETE_IMPORTED_FOLDS",
                    "Imported development assignments must contain every declared fold from zero through folds minus one.",
                )
        if len(spec.split.seeds) > 1:
            finding(
                "IMPORTED_ASSIGNMENTS_REUSED",
                "Imported memberships are identical across the selected seeds.",
                "warning",
            )
        folds = range(spec.split.folds) if imported.foldField else (None,)
        results = []
        for seed in sorted(spec.split.seeds):
            for fold in folds:
                assignment = {
                    patient: value if kind == "fixed" else "val" if value == fold else "train"
                    for patient, (kind, value) in states.items()
                }
                results.append((seed, fold, assignment))
        return results

    @staticmethod
    def _check_partition(spec, groups, seed, fold, assignment, finding):
        if set(assignment) != set(groups) or any(
            value not in PARTITIONS for value in assignment.values()
        ):
            finding(
                "INCOMPLETE_ASSIGNMENTS",
                "Every included patient must have exactly one valid partition in every seed and fold.",
            )
        result = {"seed": seed, "fold": fold}
        for partition in PARTITIONS:
            patients = [patient for patient, value in assignment.items() if value == partition]
            classes = Counter(groups[patient][0]["label"] for patient in patients)
            stats = _cohort_stats([row for patient in patients for row in groups[patient]])
            result[partition] = {
                "patients": stats["patientCount"],
                "groups": len(patients),
                "fallbackSlides": stats["fallbackSlideCount"],
                "slides": sum(len(groups[patient]) for patient in patients),
                "classes": {label: classes[label] for label in spec.target.classes},
            }
            required = partition in {"train", "val"} or bool(patients)
            if spec.split.mode == "rules":
                required = partition == "train" or bool(getattr(spec.split.rules, partition))
            if partition == "test" and spec.split.mode == "holdout" and spec.split.ratios.test > 0:
                required = True
            if required and len(patients) < spec.constraints.minPatientsPerPartition:
                finding(
                    "PARTITION_TOO_SMALL",
                    f"Seed {seed}, fold {fold}, {partition} has fewer than {spec.constraints.minPatientsPerPartition} patients.",
                )
            if required:
                for label in spec.target.classes:
                    if classes[label] < spec.constraints.minPatientsPerClass:
                        finding(
                            "PARTITION_CLASS_TOO_SMALL",
                            f"Seed {seed}, fold {fold}, {partition} has fewer than {spec.constraints.minPatientsPerClass} patients in class '{label}'.",
                        )
        if not result["test"]["groups"]:
            finding(
                "NO_TEST_SET",
                "This protocol has no held-out test patients; validation results must not be presented as held-out test performance.",
                "warning",
            )
        return result

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
            # freshness belongs to preflight, and cannot rewrite a frozen protocol.
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
        if "featurePack" in preview:
            manifest["featurePack"] = preview["featurePack"]
        return self.store.publish_configuration(
            draft_id,
            expected_revision=expected_revision,
            manifest=manifest,
            operation_id=operation_id,
            version_label=version_label,
        )
