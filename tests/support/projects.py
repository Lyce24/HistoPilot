"""Datasets, feature bundles and evaluation cohorts that many test modules build on.

Feature validation and packing queue a ``packing`` task in this test's Task Center, as in
production, and ``support.workers.run_pack`` runs its worker as the runner would.
"""

import json
from uuid import uuid4

import h5py
import numpy as np

from histopilot.application.evaluations import EvaluationService
from histopilot.application.feature_bundles import FeatureBundleService
from histopilot.application.feature_packs import FeaturePackService
from histopilot.application.features import FeatureService
from histopilot.schemas.feature_bundles import FeatureBundleSpec
from histopilot.schemas.feature_packs import FeaturePackSpec
from histopilot.schemas.features import FeatureSpec
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.io import content_hash
from histopilot.taskcenter.client import default_client
from support.workers import run_pack

TARGET = {
    "field": "label",
    "task": "binary_classification",
    "unit": "patient",
    "classes": ["low", "high"],
    "labels": {"0": "low", "1": "high"},
    "positiveClass": "high",
}


def dataset(store, operation="dataset", rows=None, inventory=None):
    """Publish a dataset of ``rows`` (four slides of four patients by default)."""
    if rows is None:
        rows = [
            {
                "slideId": f"s{i}",
                "patientId": f"p{i}",
                "attributes": {
                    "label": str(i % 2),
                    "cohort": "development" if i < 2 else "test",
                },
            }
            for i in range(4)
        ]
    draft = store.create_draft("import", operation, {})
    document = store.publish_dataset(
        draft["id"],
        expected_revision=1,
        operation_id=operation,
        manifest={
            "kind": "dataset",
            "dictionary": [
                {"key": key, "sourceColumn": key, "owner": "slide", "type": "text"}
                for key in ("label", "cohort")
            ],
        },
        artifacts={
            "records.json": json.dumps(rows).encode(),
            **({"inventory.json": json.dumps(inventory).encode()} if inventory is not None else {}),
        },
    )
    return document, rows


def packing_service(store, filesystem):
    """Feature validation and packing jobs that queue in this test's Task Center."""
    return FeaturePackService(store, filesystem, task_center=default_client())


def bundle(
    store,
    root,
    data,
    ids,
    name="features",
    dimension=4,
    encoder="uni_v1",
    pack=False,
    dtype="float32",
    feature_kind="patch",
):
    """Freeze features for ``ids`` and a bundle over them; returns (bundle, pack id, folder).

    The features are validated, or packed with ``pack=True``, by a Task Center packing task.
    """
    source = root / name
    source.mkdir()
    for identity in ids:
        with h5py.File(source / f"{identity}.h5", "w") as handle:
            handle.create_dataset(
                "features",
                data=np.ones((1 if feature_kind == "slide" else 3, dimension), dtype=dtype),
            )
            if feature_kind == "patch":
                handle.create_dataset("coords", data=np.ones((3, 2), dtype="int64"))
    filesystem = LocalFilesystem((root,))
    features = FeatureService(store, filesystem)
    spec = FeatureSpec(
        datasetId=data["id"], path=str(source), encoderId=encoder, featureKind=feature_kind
    )
    frozen = features.freeze(spec, features.preview(spec)["previewHash"], name)
    packs = packing_service(store, filesystem)
    packing = FeaturePackSpec(featureSetId=frozen["id"], action="pack" if pack else "validate")
    job = packs.submit(packing, packs.preview(packing)["previewHash"], name + "-validation")
    result = run_pack(packs.tasks.client.store, job)
    assert result["state"] == "succeeded", result
    assert packs.get(job["id"])["state"] == "succeeded"
    bundle_service = FeatureBundleService(store, filesystem)
    pack_id = result["artifact"]["id"] if pack else None
    spec = FeatureBundleSpec(
        featureSetId=frozen["id"], packArtifactIds=[pack_id] if pack_id else []
    )
    preview = bundle_service.preview(spec)
    assert preview["canFreeze"], preview["findings"]
    return bundle_service.freeze(spec, preview["previewHash"], name + "-bundle"), pack_id, source


def lifecycle(store, document, state):
    """Move a published configuration to lifecycle ``state`` (active, archived, trashed)."""
    store.lifecycle.apply(
        {f"configuration:{document['id']}": state},
        operation_id=uuid4().hex,
        request_hash=content_hash({"id": document["id"], "state": state}),
        expected_revision=store.lifecycle.read()["revision"],
    )


# -- evaluation cohorts --------------------------------------------------------------------


def setup(store, root, pack=False):
    """An evaluation service and a cohort spec for the test slides of ``dataset(store)``.

    The two development slides are the train partition of a frozen protocol; one bundle
    covers all four slides. Returns (service, spec, feature folder).
    """
    data, rows = dataset(store)
    features, pack_id, source = bundle(
        store, root, data, [row["slideId"] for row in rows], pack=pack
    )
    protocol = store.publish_configuration(
        manifest={
            "kind": "protocol",
            "datasetId": data["id"],
            "spec": {"target": TARGET, "split": {"version": 4}},
            "memberships": [
                {"slideId": row["slideId"], "patientId": row["patientId"], "partition": "train"}
                for row in rows[:2]
            ],
        },
        operation_id="protocol",
    )
    spec = {
        "protocolId": protocol["id"],
        "developmentFeatureBundleId": features["id"],
        "datasetId": data["id"],
        "featureBundleId": features["id"],
        "target": TARGET,
        "eligibility": [{"field": "cohort", "op": "eq", "value": "test"}],
    }
    if pack:
        spec["inference"] = {"loadingPolicy": "packed", "packArtifactId": pack_id}
    return EvaluationService(store, LocalFilesystem((root,))), spec, source


def draft(service, spec, name="Later cohort"):
    """An evaluation cohort draft of ``spec``."""
    return service.store.create_draft(
        "experiment", name, {"type": "evaluation-cohort", "spec": spec}
    )


def preview(service, spec):
    """The review of a new evaluation cohort draft of ``spec``."""
    item = draft(service, spec)
    return service.preview(item["id"], 1)


def codes(result):
    """The error codes among a review's findings."""
    return {item["code"] for item in result["findings"] if item["severity"] == "error"}


def slide_bundle(store, filesystem, root, dataset, rows, name):
    """Publish real validation-worker receipts, never synthesize a valid bundle."""
    source = root / name
    source.mkdir()
    random = np.random.default_rng(37)
    for index, row in enumerate(rows):
        values = random.standard_normal(8).astype("float32")
        # A single inventory mixes the two supported on-disk storage shapes.
        values = values if index % 2 else values.reshape(1, -1)
        with h5py.File(source / f"{row['slideId']}.h5", "w") as handle:
            handle.create_dataset("features", data=values)
    features = FeatureService(store, filesystem)
    spec = FeatureSpec(
        datasetId=dataset["id"],
        path=str(source),
        featureKind="slide",
        encoderId="titan",
        layout="flat",
    )
    reviewed = features.preview(spec)
    assert reviewed["canFreeze"], reviewed["findings"]
    feature = features.freeze(spec, reviewed["previewHash"], name)
    packing = packing_service(store, filesystem)
    request = FeaturePackSpec(featureSetId=feature["id"], action="validate")
    preview = packing.preview(request)
    assert preview["canRun"], preview["findings"]
    job = packing.submit(request, preview["previewHash"], name + "-validation")
    result = run_pack(packing.tasks.client.store, job)
    assert result["state"] == "succeeded", result
    assert result["validation"]["tensorValidationComplete"]
    assert result["validation"]["featureKind"] == "slide"
    bundles = FeatureBundleService(store, filesystem)
    request = FeatureBundleSpec(featureSetId=feature["id"])
    preview = bundles.preview(request)
    assert preview["canFreeze"], preview["findings"]
    return bundles.freeze(request, preview["previewHash"], name + "-bundle")
