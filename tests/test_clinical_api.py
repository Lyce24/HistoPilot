"""The clinical HTTP boundary accepts choices and exports saved server reports."""

import runpy
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from histopilot.api.clinical import clinical_router

support = runpy.run_path(str(Path(__file__).with_name("test_clinical.py")))
service = support["service"]


def test_clinical_preview_save_list_and_download_routes(service, monkeypatch):
    current, selection, *_ = service
    monkeypatch.setattr("histopilot.api.clinical.ClinicalService", lambda *_: current)
    projects = SimpleNamespace(scientific_store=lambda _identity: current.store)
    app = FastAPI()
    app.include_router(clinical_router(projects, current.filesystem))
    base = "/api/v1/projects/clinical-project/clinical-analyses"
    with TestClient(app) as client:
        assert client.get(base).json() == {"items": []}
        result = client.post(base + "/preview", json=selection.model_dump())
        assert result.status_code == 200
        preview = result.json()
        assert preview["canSave"]
        result = client.post(
            base,
            json={
                **selection.model_dump(),
                "previewHash": preview["previewHash"],
                "operationId": "api-report",
            },
        )
        assert result.status_code == 201
        saved = result.json()
        assert client.get(base + "/" + saved["id"]).json() == saved
        assert client.get(base).json() == {"items": [saved]}
        exported = client.get(base + "/" + saved["id"] + "/artifacts/report.json")
        assert exported.status_code == 200
        assert exported.json() == saved
        assert exported.headers["content-disposition"] == 'attachment; filename="report.json"'
        csv = client.get(base + "/" + saved["id"] + "/artifacts/operating-curves.csv")
        assert csv.status_code == 200
        assert "netInterventionsAvoidedPer100" in csv.text.splitlines()[0]
        assert (
            client.post(
                base + "/preview", json={**selection.model_dump(), "probabilities": []}
            ).status_code
            == 422
        )
        assert (
            client.post(
                base + "/preview", json={**selection.model_dump(), "threshold": 1}
            ).status_code
            == 422
        )
