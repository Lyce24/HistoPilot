"""Interpretation reuses its selected bundle's frozen dataset folder automatically."""

import json
from pathlib import Path

import pytest
from test_interpretation import study as study
from test_interpretation_gallery import gallery as gallery

from histopilot.schemas.interpretation import InterpretationGalleryQuery, VisualizeInterpretation
from histopilot.storage.project_lock import StorageError


def dataset(service, operation, records, *, manifest=None):
    draft = service.store.create_draft("import", operation, {})
    return service.store.publish_dataset(
        draft["id"],
        expected_revision=1,
        manifest={"kind": "dataset", "name": operation, **(manifest or {})},
        artifacts={"records.json": json.dumps(records).encode()},
        operation_id=operation,
    )


def test_sources_reuse_external_bundle_dataset_not_predictor_training_folder(gallery):
    service, source, _, _ = gallery
    returned = service.gallery.sources()["items"][0]
    bundle = service.gallery.bundles.get(source["featureBundleId"])
    predictor = service.predictors.get(source["predictorId"])
    assert returned["datasetId"] == bundle["manifest"]["datasetId"]
    assert returned["datasetId"] != predictor["manifest"]["datasetId"]
    assert returned["datasetName"] == "Gallery"
    assert returned["slideFolder"] == source["slideFolder"]
    assert returned["slideFolderSource"] == "dataset_records"
    assert returned["slideFolderFinding"] is None and returned["current"]


def test_gallery_and_visualization_can_omit_previously_saved_folder(gallery):
    service, source, executor, _ = gallery
    automatic = {key: value for key, value in source.items() if key != "slideFolder"}
    results = service.gallery.query(InterpretationGalleryQuery(**automatic, search="tumour"))
    assert results["folder"] == source["slideFolder"]
    assert results["total"] == 1 and results["items"][0]["available"]
    request = VisualizeInterpretation(
        **automatic, slidePaths=[results["items"][0]["slidePath"]], operationId="automatic-folder"
    )
    visualized = service.visualize(request)
    assert visualized["items"][0]["status"] == "queued", visualized
    assert visualized["interpretations"][0]["manifest"]["slideFolder"] == source["slideFolder"]
    assert service.visualize(request)["items"][0]["reused"]
    assert len(executor.calls) == 1


@pytest.mark.parametrize("location", ["provenance", "spec"])
def test_saved_dataset_import_folder_supports_nested_record_folders(gallery, tmp_path, location):
    service, _, _, _ = gallery
    folder = tmp_path / "frozen-external-slides"
    (folder / "cohort-a").mkdir(parents=True)
    (folder / "cohort-b").mkdir()
    records = [
        {"slideId": name, "slidePath": str(folder / group / f"{name}.png")}
        for name, group in [("a", "cohort-a"), ("b", "cohort-b")]
    ]
    manifest = (
        {"provenance": {"mapping": {"slideRoot": str(folder)}}}
        if location == "provenance"
        else {"spec": {"slideRoot": str(folder)}}
    )
    frozen = dataset(service, f"Declared {location}", records, manifest=manifest)
    resolved = service.gallery.dataset_source(frozen["id"])
    assert resolved["slideFolder"] == str(folder)
    assert resolved["slideFolderSource"] == "dataset_import"
    assert resolved["slideFolderFinding"] is None


def test_legacy_common_ancestor_is_not_guessed_for_unrelated_folders(gallery, tmp_path):
    service, _, _, _ = gallery
    first, second = tmp_path / "cohort-one", tmp_path / "cohort-two"
    first.mkdir()
    second.mkdir()
    frozen = dataset(
        service,
        "Ambiguous folders",
        [
            {"slideId": "a", "slidePath": str(first / "a.png")},
            {"slideId": "b", "slidePath": str(second / "b.png")},
        ],
    )
    resolved = service.gallery.dataset_source(frozen["id"])
    assert resolved["slideFolder"] is None
    assert resolved["slideFolderSource"] is None
    assert "multiple folders" in resolved["slideFolderFinding"]["message"]
    assert "Open Datasets" in resolved["slideFolderFinding"]["message"]


@pytest.mark.parametrize(
    "scenario", ["no_paths", "outside_root", "symlink", "missing_folder", "traversal"]
)
def test_missing_or_unsafe_dataset_source_requires_dataset_repair(gallery, tmp_path, scenario):
    service, _, _, _ = gallery
    folder = tmp_path / "unsafe-source"
    folder.mkdir()
    path = folder / "a.png"
    manifest = {}
    if scenario == "no_paths":
        records = [{"slideId": "a", "slidePath": None}]
    else:
        if scenario == "outside_root":
            path = Path("/etc/a.png")
        elif scenario == "symlink":
            link = tmp_path / "linked-source"
            link.symlink_to(folder, target_is_directory=True)
            path = link / "a.png"
        elif scenario == "missing_folder":
            path = tmp_path / "deleted-source" / "a.png"
            manifest = {"provenance": {"mapping": {"slideRoot": str(path.parent)}}}
        elif scenario == "traversal":
            path = folder / ".." / "outside.png"
        records = [{"slideId": "a", "slidePath": str(path)}]
    frozen = dataset(service, f"Invalid {scenario}", records, manifest=manifest)
    resolved = service.gallery.dataset_source(frozen["id"])
    assert resolved["slideFolder"] is None
    assert resolved["slideFolderFinding"]["code"] == "INTERPRETATION_DATASET_FOLDER_UNAVAILABLE"
    assert "Open Datasets" in resolved["slideFolderFinding"]["message"]


def test_omitted_folder_unavailable_is_clear_but_explicit_legacy_override_still_works(
    gallery, monkeypatch
):
    service, source, _, _ = gallery
    original = service.gallery.dataset_source
    metadata = original(service.gallery.sources()["items"][0]["datasetId"])
    finding = {
        "severity": "error",
        "code": "INTERPRETATION_DATASET_FOLDER_UNAVAILABLE",
        "message": "Open Datasets to link the slide folder.",
    }
    monkeypatch.setattr(
        service.gallery,
        "dataset_source",
        lambda *args, **kwargs: {
            **metadata,
            "slideFolder": None,
            "slideFolderSource": None,
            "slideFolderFinding": finding,
        },
    )
    automatic = {key: value for key, value in source.items() if key != "slideFolder"}
    with pytest.raises(StorageError, match="Open Datasets"):
        service.gallery.query(InterpretationGalleryQuery(**automatic))
    assert service.gallery.query(InterpretationGalleryQuery(**source))["total"] == 3
    item = service.gallery.sources()["items"][0]
    assert not item["current"] and item["slideFolderFinding"] == finding
    assert finding in item["findings"]


def test_automatic_folder_gallery_http_auth_and_request(gallery, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from histopilot.api.app import create_app
    from histopilot.config import Settings

    service, source, executor, _ = gallery
    monkeypatch.setattr(
        "histopilot.api.interpretation.InterpretationService", lambda *args: service
    )
    app = create_app(Settings(workspace=tmp_path / "automatic-registry", data_roots=(tmp_path,)))
    monkeypatch.setattr(app.state.projects, "scientific_store", lambda identity: service.store)
    automatic = {key: value for key, value in source.items() if key != "slideFolder"}
    base = "/api/v1/projects/test/interpretations"
    with TestClient(app, base_url="http://127.0.0.1:8787") as client:
        assert client.post(base + "/gallery", json=automatic).status_code == 401
        client.headers["X-HistoPilot-Token"] = client.get("/api/v1/session").json()["token"]
        response = client.post(base + "/gallery", json={**automatic, "search": "001"})
        assert response.status_code == 200, response.text
        path = response.json()["items"][0]["slidePath"]
        response = client.post(
            base + "/visualize",
            json={**automatic, "slidePaths": [path], "operationId": "api-automatic"},
        )
        assert response.status_code == 202, response.text
        assert response.json()["items"][0]["status"] == "queued"
        assert len(executor.calls) == 1
