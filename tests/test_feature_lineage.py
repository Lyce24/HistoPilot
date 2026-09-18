"""An attached feature version preserves evidence from its actual extraction job."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from histopilot.application.features import FeatureService
from histopilot.schemas.features import FeatureSpec
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore


@pytest.fixture
def extracted(tmp_path):
    folder = tmp_path / "project"
    folder.mkdir()
    store = ScientificStore(folder, "project-lineage")
    draft = store.create_draft("import", "slides", {})
    dataset = store.publish_dataset(
        draft["id"],
        expected_revision=1,
        manifest={"kind": "dataset"},
        artifacts={"records.json": b'[{"slideId":"001.A"}]'},
        operation_id="dataset",
    )
    features = folder / "trident" / "20x_256px_0px_overlap" / "features_uni_v1"
    features.mkdir(parents=True)
    with h5py.File(features / "001.A.h5", "w") as handle:
        handle.create_dataset("features", data=np.ones((3, 4), dtype="float32"))
        handle.create_dataset("coords", data=np.arange(6, dtype="int64").reshape(3, 2))
    identity = "extraction-" + "a" * 32
    job_dir = folder / "extractions" / identity
    job_dir.mkdir(parents=True)
    job = {
        "id": identity,
        "slideCount": 1,
        "spec": {"datasetId": dataset["id"], "options": {"task": "all"}},
        "outputLayout": {"featuresDir": str(features), "featureKind": "patch"},
        "runtime": {"pythonPath": "/usr/bin/python3", "scriptSha256": "source-evidence"},
        "command": ["python", "run_batch_of_slides.py"],
        "inputFiles": [],
    }
    (job_dir / "job.json").write_text(json.dumps(job))
    (job_dir / "result.json").write_text(json.dumps({"state": "succeeded"}))
    (job_dir / "validation.json").write_text(
        json.dumps(
            {
                "jobId": identity,
                "completedSlides": 1,
                "missingSlides": 0,
                "unvalidatedSlides": 0,
            }
        )
    )
    spec = FeatureSpec(datasetId=dataset["id"], path=str(features), sourceExtractionJobId=identity)
    return FeatureService(store, LocalFilesystem((tmp_path,))), spec, job_dir


def test_extraction_evidence_is_frozen_and_later_changes_are_detected(extracted):
    service, spec, job_dir = extracted
    preview = service.preview(spec)
    assert preview["canFreeze"]
    assert preview["sourceExtraction"]["jobId"] == spec.sourceExtractionJobId
    feature = service.freeze(spec, preview["previewHash"], "feature")
    assert service.verify_binding(feature) == []
    job = json.loads((job_dir / "job.json").read_text())
    job["runtime"]["scriptSha256"] = "different-source"
    (job_dir / "job.json").write_text(json.dumps(job))
    assert any(row["code"] == "FEATURE_SOURCE_CHANGED" for row in service.verify_binding(feature))


def test_an_extraction_without_a_dataset_can_still_supply_provenance(extracted):
    """Either side may name no cohort; the job, encoder and output folder still pin the run."""
    service, spec, job_dir = extracted
    job = json.loads((job_dir / "job.json").read_text())
    job["spec"]["datasetId"] = None
    (job_dir / "job.json").write_text(json.dumps(job))
    preview = service.preview(spec)
    assert preview["sourceExtraction"]["jobId"] == job["id"]
    assert preview["sourceExtraction"]["spec"]["datasetId"] is None


@pytest.mark.parametrize(
    "change",
    [
        "folder",
        "failed",
        "incomplete",
        "slide",
        "segmentation",
        "encoder",
        "inspection",
        "finding",
    ],
)
def test_unrelated_or_incomplete_extraction_cannot_supply_provenance(extracted, change):
    service, spec, job_dir = extracted
    job = json.loads((job_dir / "job.json").read_text())
    if change == "folder":
        job["outputLayout"]["featuresDir"] = str(job_dir)
    elif change == "failed":
        (job_dir / "result.json").write_text('{"state":"failed"}')
    elif change == "incomplete":
        (job_dir / "validation.json").write_text(
            json.dumps(
                {
                    "jobId": job["id"],
                    "completedSlides": 0,
                    "missingSlides": 1,
                }
            )
        )
    elif change == "slide":
        job["outputLayout"]["featureKind"] = "slide"
    elif change == "segmentation":
        job["spec"]["options"]["task"] = "seg"
    elif change == "encoder":
        job["spec"]["options"]["patch_encoder"] = "virchow"
    elif change in {"inspection", "finding"}:
        path = job_dir / "validation.json"
        validation = json.loads(path.read_text())
        if change == "inspection":
            validation["inspectionComplete"] = False
        else:
            validation["findings"] = [{"severity": "error", "code": "INVALID_EXTRACTION_ARTIFACT"}]
        path.write_text(json.dumps(validation))
    (job_dir / "job.json").write_text(json.dumps(job))
    with pytest.raises(StorageError, match="linked extraction"):
        service.preview(spec)


def test_external_attachment_still_works_without_inventing_job_evidence(extracted):
    service, spec, _ = extracted
    preview = service.preview(spec.model_copy(update={"sourceExtractionJobId": None}))
    assert preview["canFreeze"]
    assert "sourceExtraction" not in preview


def test_inferred_flat_encoder_must_match_the_linked_extraction(extracted):
    service, spec, _ = extracted
    with h5py.File(Path(spec.path) / "001.A.h5", "r+") as handle:
        handle["features"].attrs["encoder"] = "virchow"
    with pytest.raises(StorageError) as caught:
        service.preview(spec.model_copy(update={"layout": "flat"}))
    assert caught.value.code == "INVALID_SOURCE_EXTRACTION"


def test_completed_extraction_can_supply_a_different_dataset_revision(extracted):
    service, original, job_dir = extracted
    draft = service.store.create_draft("import", "Revised cohort", {})
    revised = service.store.publish_dataset(
        draft["id"],
        expected_revision=1,
        manifest={"kind": "dataset"},
        artifacts={"records.json": b'[{"slideId":"001.A","patientId":"reviewed-patient"}]'},
        operation_id="revised-cohort",
    )
    selected = original.model_copy(update={"datasetId": revised["id"]})
    reviewed = service.preview(selected)
    assert reviewed["canFreeze"], reviewed["findings"]
    assert [row["slideId"] for row in reviewed["files"]] == ["001.A"]
    assert reviewed["sourceExtraction"]["spec"]["datasetId"] == original.datasetId
    frozen = service.freeze(selected, reviewed["previewHash"], "reused-extraction")
    assert frozen["manifest"]["datasetId"] == revised["id"]
    assert service.verify_binding(frozen) == []
    # Cohort selection is independent, while the original run identity stays frozen.
    job = json.loads((job_dir / "job.json").read_text())
    job["spec"]["datasetId"] = revised["id"]
    (job_dir / "job.json").write_text(json.dumps(job))
    assert any(row["code"] == "FEATURE_SOURCE_CHANGED" for row in service.verify_binding(frozen))
