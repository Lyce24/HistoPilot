"""Resolve clinical predictors from authenticated immutable dataset artifacts."""

from histopilot.application.protocols import FilterEvaluator, ProtocolService
from histopilot.clinical_features import clinical_fields, clinical_rows
from histopilot.storage.project_lock import StorageError


def frozen_clinical_values(store, filesystem, dataset_ids, memberships, fields):
    if not fields:
        return {}
    selected = {row["slideId"] for row in memberships}
    values = {}
    service = ProtocolService(store, filesystem)
    names = {item["field"] for item in fields}
    for dataset_id in dataset_ids:
        _dataset, dictionary, rows = service._load_dataset(dataset_id)
        relevant = [row for row in rows if row["slideId"] in selected]
        if not relevant:
            continue
        absent = names - dictionary.keys()
        if absent:
            raise StorageError(
                "Map the required clinical fields in the frozen dataset: "
                + ", ".join(sorted(absent)),
                "CLINICAL_FIELDS_MISSING",
                422,
            )
        for row in relevant:
            if row["slideId"] in values:
                raise StorageError(
                    "Clinical slide identity occurs in multiple datasets.",
                    "CLINICAL_IDENTITY_AMBIGUOUS",
                    422,
                )
            values[row["slideId"]] = {
                name: FilterEvaluator.field(row, name) for name in sorted(names)
            }
    if set(values) != selected:
        raise StorageError(
            "Clinical values must cover every frozen selected slide.",
            "CLINICAL_COVERAGE_MISSING",
            422,
        )
    return values


# How a dictionary column becomes a covariate by default; dates are not offered.
DEFAULT_KINDS = {
    "integer": "numeric",
    "decimal": "numeric",
    "boolean": "categorical",
    "categorical": "categorical",
    "ordered_categorical": "categorical",
    "text": "categorical",
}


def _condition_fields(conditions):
    for condition in conditions or []:
        if isinstance(condition, dict) and "conditions" in condition:
            yield from _condition_fields(condition["conditions"])
        elif isinstance(condition, dict) and condition.get("field"):
            yield condition["field"]


def excluded_clinical_fields(store, protocol):
    """Fields that define the label or the split, each with the reason it is refused."""
    spec = protocol["spec"]
    excluded = {spec["target"].get("field"): "It is the prediction target."}
    source_id = spec.get("sourceTargetSplitId")
    if source_id:
        source = store.get_configuration(source_id)["manifest"]["spec"]
        testing = (source.get("testTarget") or {}).get("field")
        if testing:
            excluded.setdefault(testing, "It is the testing target.")
        split = source.get("split") or {}
        reason = "It defines the training and testing split."
        fields = [split.get("partitionField"), *_condition_fields(split.get("trainRules"))]
        fields += list(_condition_fields(split.get("testRules")))
        for field in fields:
            if field:
                excluded.setdefault(field, reason)
    return excluded


def clinical_field_choices(store, filesystem, protocol):
    """The frozen dataset's columns a model may take as clinical inputs."""
    _dataset, dictionary, _rows = ProtocolService(store, filesystem)._load_dataset(
        protocol["datasetId"]
    )
    excluded = excluded_clinical_fields(store, protocol)
    return [
        {
            "field": key,
            "owner": column.get("owner", "slide"),
            "type": column.get("type", "text"),
            "kind": DEFAULT_KINDS[column.get("type", "text")],
            "excluded": excluded.get(key),
        }
        for key, column in dictionary.items()
        if column.get("type", "text") in DEFAULT_KINDS
    ]


def development_clinical_values(store, filesystem, protocol, recipes):
    if not any(clinical_fields(recipe) for recipe in recipes):
        return {}
    choices = {row["field"]: row for row in clinical_field_choices(store, filesystem, protocol)}
    unit = protocol["spec"].get("splitUnit", "patient")
    fields = {}
    for recipe in recipes:
        for item in clinical_fields(recipe):
            choice = choices.get(item["field"])
            if choice is None:
                raise StorageError(
                    f"{item['field']} is not a numeric or categorical column of the frozen dataset.",
                    "CLINICAL_FIELD_UNKNOWN",
                    422,
                )
            if choice["excluded"]:
                raise StorageError(
                    f"{item['field']} cannot be a clinical input. {choice['excluded']}",
                    "CLINICAL_TARGET_LEAKAGE",
                    422,
                )
            fields[item["field"]] = item
    memberships = list({row["slideId"]: row for row in protocol["memberships"]}.values())
    values = frozen_clinical_values(
        store, filesystem, [protocol["datasetId"]], memberships, list(fields.values())
    )
    for recipe in recipes:
        if clinical_fields(recipe):
            try:
                clinical_rows(memberships, values, clinical_fields(recipe), unit=unit)
            except ValueError as error:
                raise StorageError(str(error), "CLINICAL_VALUES_INVALID", 422) from error
    return values
