"""Exact frozen train/test membership shared by training and evaluation derivations."""

from histopilot.storage.project_lock import StorageError


def restrict_target_split_rows(
    store, identity, dataset_id, target, rows, partition, *, split_unit="patient"
):
    source = store.get_configuration(identity)
    manifest = source["manifest"]
    if manifest.get("kind") != "target-split" or manifest.get("datasetId") != dataset_id:
        raise StorageError(
            "Targets & Splits must belong to the selected dataset.",
            "TARGET_SPLIT_DATASET_MISMATCH",
            422,
        )
    target_values = target.model_dump(mode="json") if target is not None else None
    source_spec = manifest["spec"]
    if split_unit != source_spec.get("splitUnit", "patient"):
        raise StorageError(
            "Use the slide or patient split unit frozen in Targets & Splits.",
            "TARGET_SPLIT_UNIT_MISMATCH",
            422,
        )
    expected_target = (
        source_spec.get("testTarget", source_spec["target"])
        if partition == "test"
        else source_spec["target"]
    )
    if target_values != expected_target:
        raise StorageError(
            "Use the target frozen in Targets & Splits.", "TARGET_SPLIT_TARGET_MISMATCH", 422
        )
    memberships = manifest["memberships"]
    selected = {row["slideId"] for row in memberships if row["partition"] == partition}
    if not selected:
        raise StorageError(
            f"Targets & Splits has no {partition} slides.", "EMPTY_TARGET_SPLIT_PARTITION", 422
        )
    if len({row["slideId"] for row in memberships}) != len(memberships):
        raise StorageError(
            "Frozen training and testing must contain unique slides.",
            "TARGET_SPLIT_SLIDE_LEAKAGE",
            422,
        )
    if split_unit == "patient":
        patient_roles = {}
        for row in memberships:
            patient = row["patientId"]
            if patient_roles.setdefault(patient, row["partition"]) != row["partition"]:
                raise StorageError(
                    "Frozen training and testing share patient identities.",
                    "TARGET_SPLIT_PATIENT_LEAKAGE",
                    422,
                )
    result = [row for row in rows if row["slideId"] in selected]
    if {row["slideId"] for row in result} != selected:
        raise StorageError(
            "Frozen partition slides are missing from the dataset.",
            "TARGET_SPLIT_SOURCE_CHANGED",
            409,
        )
    return result
