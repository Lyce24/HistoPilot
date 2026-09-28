"""Disposable authenticated production API fixture for real-slide viewer verification.

No network listener is opened. Existing slides are read only; all project/database
artifacts live in a TemporaryDirectory and are removed by the returned cleanup.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from histopilot.api import create_app
from histopilot.application.morphology import _image_fingerprint
from histopilot.config import Settings
from histopilot.storage.packed import _stamp
from histopilot.viewer.slide_images import _source_identity


def create_real_slide_api(catalog, metadata_records=1111):
    """Return (client, project_id, dataset_ids_by_index, token, cleanup).

    All catalog slides share one frozen dataset. Their IDs are
    ``real-slide-{slideIndex}``. Additional records are unread synthetic metadata
    placeholders, used only to exercise realistic manifest validation costs.
    """
    if isinstance(catalog, (str, Path)):
        catalog = json.loads(Path(catalog).read_text())
    slides = catalog["slides"]
    if not slides or len(slides) > 5:
        raise ValueError("Choose a catalog with 1–5 real slide images")
    if not len(slides) <= metadata_records <= 5000:
        raise ValueError("The metadata workload must contain the slides and at most 5000 records")
    resources = ExitStack()
    try:
        root = Path(resources.enter_context(TemporaryDirectory(prefix="histopilot-real-api-")))
        paths = [Path(slide["path"]).absolute() for slide in slides]
        settings = Settings(
            workspace=root / "workspace", data_roots=(root, *(path.parent for path in paths))
        )
        app = create_app(settings)
        client = resources.enter_context(TestClient(app, base_url="http://127.0.0.1:8787"))
        response = client.get("/api/v1/session")
        response.raise_for_status()
        token = response.json()["token"]
        client.headers["X-HistoPilot-Token"] = token
        response = client.post(
            "/api/v1/projects",
            json={
                "name": "Temporary real-slide viewer verification",
                "storagePath": str(root / "project"),
            },
        )
        response.raise_for_status()
        project_id = response.json()["id"]
        store = app.state.projects.scientific_store(project_id)
        records, inventory = [], []
        for slide, path in zip(slides, paths, strict=True):
            stamp = _stamp(path.stat())
            if _image_fingerprint(path, stamp) != slide["sourceFingerprint"] or (
                "sourceStamp" in slide and list(_source_identity(path)) != slide["sourceStamp"]
            ):
                raise ValueError("A catalog slide changed before API fixture preparation")
            identity = f"real-slide-{slide['slideIndex']}"
            records.append(
                {"slideId": identity, "patientId": None, "slidePath": str(path), "attributes": {}}
            )
            inventory.append({"slideId": identity, "path": str(path), **stamp})
        for index in range(len(slides), metadata_records):
            identity = f"unread-placeholder-{index:04d}"
            path = str(root / "unread-images" / f"{identity}.svs")
            records.append(
                {"slideId": identity, "patientId": None, "slidePath": path, "attributes": {}}
            )
            inventory.append({"slideId": identity, "path": path, **stamp})
        draft = store.create_draft("import", "Temporary image-validation workload", {})
        dataset = store.publish_dataset(
            draft["id"],
            expected_revision=1,
            manifest={"kind": "dataset"},
            artifacts={
                "records.json": json.dumps(records).encode(),
                "inventory.json": json.dumps(inventory).encode(),
            },
            operation_id="real-slide-viewer-fixture",
        )
        datasets = {slide["slideIndex"]: dataset["id"] for slide in slides}
        return client, project_id, datasets, token, resources.close
    except BaseException:
        resources.close()
        raise
