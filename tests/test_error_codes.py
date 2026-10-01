"""The error-code registry matches the raise sites, its page, and the service's error bodies."""

import ast
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from histopilot.api.app import create_app
from histopilot.api.error_codes import ALIASES, COMPUTED, ERROR_CODES, KINDS, kind_for
from histopilot.application.imports import ImportService
from histopilot.config import Settings
from histopilot.schemas.imports import ImportSpec
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore

ROOT = Path(__file__).resolve().parents[1]
API = "/api/v1"
generator = runpy.run_path(str(ROOT / "scripts" / "error_codes.py"))
SCAN = generator["scan"]()
RAISED = SCAN.codes()
LISTED = {code for codes in COMPUTED.values() for code in codes}
REGENERATE = "Run python scripts/error_codes.py after registering codes."


def test_every_raised_code_is_registered():
    missing = {
        code: [f"{site.path}:{site.line}" for site in sites[:3]]
        for code, sites in RAISED.items()
        if code not in ERROR_CODES
    }
    assert not missing, f"Add these to ERROR_CODES in histopilot/api/error_codes.py: {missing}"


def test_every_registered_code_is_still_raised():
    stale = set(ERROR_CODES) - set(RAISED) - LISTED - set(ALIASES)
    assert not stale, f"No raise site names these; remove them from ERROR_CODES: {sorted(stale)}"


def test_kinds_are_the_contracts():
    assert KINDS == ("invalid", "refused", "conflict", "not-found", "unavailable", "internal")
    unknown = {code: kind for code, kind in ERROR_CODES.items() if kind not in KINDS}
    assert not unknown


def test_runtime_codes_are_listed_under_the_module_that_raises_them():
    unlisted = [
        f"{item.path}:{item.line} {item.expression}"
        for item in SCAN.computed
        if item.path not in COMPUTED
    ]
    assert not unlisted, f"List the codes these can raise in COMPUTED: {unlisted}"
    raising = {item.path for item in SCAN.computed}
    assert set(COMPUTED) <= raising, "COMPUTED names a module without a runtime code"
    assert LISTED <= set(ERROR_CODES)
    # A listed code must still exist somewhere, even if only in a finding.
    literals = {
        node.value
        for module in generator["_modules"]()
        for node in ast.walk(module.tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert not LISTED - literals, f"No longer in histopilot/: {sorted(LISTED - literals)}"


def test_aliases_keep_their_replacements_kind():
    assert ALIASES == {"STALE_PREVIEW": "PREVIEW_STALE"}
    for alias, replacement in ALIASES.items():
        assert ERROR_CODES[alias] == ERROR_CODES[replacement] == "conflict"


def test_every_exception_with_a_code_is_scanned():
    carriers = set()
    for module in generator["_modules"]():
        for node in ast.walk(module.tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in ast.walk(node):
                if (
                    isinstance(item, ast.Attribute)
                    and isinstance(item.ctx, ast.Store)
                    and isinstance(item.value, ast.Name)
                    and item.value.id == "self"
                    and item.attr == "code"
                ):
                    carriers.add(node.name)
    assert carriers <= set(generator["CARRIERS"]), "Add the class to CARRIERS in the generator"


def test_the_page_matches_the_generator():
    page = (ROOT / "docs" / "error-codes.md").read_text(encoding="utf-8")
    assert page == generator["render"](SCAN), REGENERATE
    assert set(generator["MEANINGS"]) <= set(ERROR_CODES)
    assert "/home/" not in page and "/mnt/" not in page


def test_the_registry_imports_only_the_standard_library():
    script = """
import sys
import histopilot.api.error_codes
loaded = sorted(
    name for name in ("fastapi", "starlette", "pydantic", "histopilot.api.app") if name in sys.modules
)
if loaded:
    raise SystemExit("the registry imported " + ", ".join(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    "code,status,kind",
    [
        ("PROJECT_BUSY", 500, "conflict"),  # The registry wins over the status.
        ("STALE_PREVIEW", 409, "conflict"),
        ("SOMETHING_NEW", 400, "invalid"),
        ("SOMETHING_NEW", 422, "invalid"),
        ("SOMETHING_NEW", 403, "refused"),
        ("SOMETHING_NEW", 413, "refused"),
        ("SOMETHING_NEW", 404, "not-found"),
        ("OUTPUT_NEW_BUSY", 409, "conflict"),
        ("SPEC_STALE", 409, "conflict"),
        ("NEW_REVISION_CONFLICT", 409, "conflict"),
        ("SOMETHING_NEW", 409, "refused"),
        ("SOMETHING_NEW", 503, "conflict"),
        ("SOMETHING_NEW", 504, "conflict"),
        ("SOMETHING_NEW", 500, "internal"),
        ("SOMETHING_NEW", 502, "internal"),
        ("SOMETHING_NEW", 401, "unavailable"),
        (None, 422, "invalid"),
        (None, None, "internal"),
    ],
)
def test_kind_for_uses_the_registry_then_the_contracts_status_rules(code, status, kind):
    assert kind_for(code, status) == kind


@pytest.fixture
def client(tmp_path):
    settings = Settings(workspace=tmp_path / "registry", data_roots=(tmp_path,))
    with TestClient(create_app(settings), base_url="http://127.0.0.1:8787") as test_client:
        test_client.headers["X-HistoPilot-Token"] = test_client.get(f"{API}/session").json()[
            "token"
        ]
        yield test_client


@pytest.mark.parametrize(
    "headers,status,code",
    [
        ({"Host": "evil.example:8787"}, 400, "HOST_REFUSED"),
        ({"Origin": "http://localhost:9999"}, 403, "ORIGIN_REFUSED"),
        ({"Sec-Fetch-Site": "cross-site"}, 403, "CROSS_SITE_REFUSED"),
        ({"Sec-Fetch-Site": "sideways"}, 403, "BROWSER_CONTEXT_REFUSED"),
        ({"Sec-Fetch-Site": "same-site"}, 403, "ORIGIN_REQUIRED"),
        ({"X-HistoPilot-Token": "wrong"}, 401, "SESSION_TOKEN_REQUIRED"),
    ],
)
def test_boundary_refusals_carry_codes(client, headers, status, code):
    response = client.get(f"{API}/projects", headers=headers)
    assert response.status_code == status
    body = response.json()
    assert body["code"] == code and ERROR_CODES[code] == "unavailable"
    assert isinstance(body["detail"], str) and body["detail"]


def test_workspace_filesystem_and_route_errors_carry_codes(client):
    missing = client.get(f"{API}/projects/project-{'0' * 32}/workspace")
    assert missing.status_code == 404
    assert missing.json() == {
        "detail": "The project does not exist. Load its folder first.",
        "code": "PROJECT_NOT_FOUND",
    }
    relative = client.get(f"{API}/filesystem/list", params={"path": "relative"})
    assert relative.status_code == 400
    assert relative.json()["code"] == "PATH_NOT_ABSOLUTE"
    unknown = client.get(f"{API}/no-such-route")
    assert unknown.status_code == 404
    assert unknown.json() == {"detail": "Unknown API endpoint.", "code": "API_ENDPOINT_UNKNOWN"}
    wrong_method = client.post(f"{API}/no-such-route", json={})
    assert wrong_method.status_code == 405
    assert wrong_method.json()["code"] == "API_METHOD_NOT_ALLOWED"
    # Static assets keep FastAPI's plain body.
    asset = client.get("/assets/missing.js")
    assert asset.status_code == 404 and asset.json() == {"detail": "Static asset not found."}


def test_schema_errors_keep_their_field_list_and_add_a_code(client):
    response = client.post(f"{API}/projects", json={})
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "REQUEST_INVALID"
    assert isinstance(body["detail"], list) and body["detail"]
    assert {"loc", "msg", "type"} <= set(body["detail"][0])


def test_unexpected_errors_are_json_and_still_raised_for_the_server_log(tmp_path, monkeypatch):
    settings = Settings(workspace=tmp_path / "registry", data_roots=(tmp_path,))
    app = create_app(settings)

    def broken():
        raise RuntimeError("private detail that must not reach the client")

    monkeypatch.setattr(app.state.projects, "list_projects", broken)
    with TestClient(
        app, base_url="http://127.0.0.1:8787", raise_server_exceptions=False
    ) as test_client:
        token = test_client.get(f"{API}/session").json()["token"]
        response = test_client.get(f"{API}/projects", headers={"X-HistoPilot-Token": token})
    assert response.status_code == 500
    assert response.json()["code"] == "INTERNAL_ERROR"
    assert "private detail" not in response.text
    # Starlette raises the error again after responding, so the server logs its traceback.
    with TestClient(app, base_url="http://127.0.0.1:8787") as test_client:
        token = test_client.get(f"{API}/session").json()["token"]
        with pytest.raises(RuntimeError):
            test_client.get(f"{API}/projects", headers={"X-HistoPilot-Token": token})


def test_storage_errors_send_their_findings(tmp_path, monkeypatch):
    settings = Settings(workspace=tmp_path / "registry", data_roots=(tmp_path,))
    app = create_app(settings)
    finding = {"code": "SLIDE_SOURCE_REQUIRED", "message": "Choose.", "severity": "error"}

    def blocked():
        raise StorageError(
            "Resolve the blocking import findings before freezing.",
            "IMPORT_BLOCKED",
            422,
            findings=[{**finding, "examples": ["slide-1"], "count": 1}],
        )

    monkeypatch.setattr(app.state.projects, "list_projects", blocked)
    with TestClient(app, base_url="http://127.0.0.1:8787") as test_client:
        token = test_client.get(f"{API}/session").json()["token"]
        response = test_client.get(f"{API}/projects", headers={"X-HistoPilot-Token": token})
    assert response.status_code == 422
    assert response.json() == {
        "detail": "Resolve the blocking import findings before freezing.",
        "code": "IMPORT_BLOCKED",
        "findings": [finding],
    }


def test_a_blocked_freeze_names_the_findings_that_blocked_it(tmp_path):
    project, sources = tmp_path / "project", tmp_path / "sources"
    project.mkdir()
    sources.mkdir()
    store = ScientificStore(project, "project-error-codes")
    store.initialize()
    service = ImportService(store, LocalFilesystem((sources,)))
    (sources / "metadata.csv").write_text("Slide_ID,Label\n001,NA\n", encoding="utf-8")
    spec = ImportSpec.model_validate(
        {
            "source": {"path": str(sources / "metadata.csv")},
            "slideIdColumn": "Slide_ID",
            "includeMissingSlides": False,
        }
    )
    saved = store.create_draft(
        "import", "Labels", {"type": "dataset-import", "spec": spec.model_dump()}
    )
    reviewed = service.preview(saved["id"], saved["revision"])
    assert not reviewed["canFreeze"]
    with pytest.raises(StorageError) as error:
        service.freeze(saved["id"], saved["revision"], reviewed["previewHash"], "freeze")
    assert error.value.code == "IMPORT_BLOCKED"
    assert error.value.findings == reviewed["findings"]
    assert "SLIDE_SOURCE_REQUIRED" in {item["code"] for item in error.value.findings}
