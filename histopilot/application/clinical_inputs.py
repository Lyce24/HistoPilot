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


def development_clinical_values(store, filesystem, protocol, recipes):
    fields = {}
    allowed = set(protocol["spec"].get("predictors", []))
    for recipe in recipes:
        for item in clinical_fields(recipe):
            if item["field"] not in allowed:
                raise StorageError(
                    f"Declare {item['field']} as an extra spreadsheet input in Targets & splits before using it.",
                    "CLINICAL_FIELD_NOT_DECLARED",
                    422,
                )
            if item["field"] == protocol["spec"]["target"]["field"]:
                raise StorageError(
                    "The target cannot be a clinical predictor.", "CLINICAL_TARGET_LEAKAGE", 422
                )
            fields[item["field"]] = item
    memberships = list({row["slideId"]: row for row in protocol["memberships"]}.values())
    values = frozen_clinical_values(
        store, filesystem, [protocol["datasetId"]], memberships, list(fields.values())
    )
    for recipe in recipes:
        if clinical_fields(recipe):
            try:
                clinical_rows(memberships, values, clinical_fields(recipe))
            except ValueError as error:
                raise StorageError(str(error), "CLINICAL_VALUES_INVALID", 422) from error
    return values
