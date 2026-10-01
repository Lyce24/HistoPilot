"""Every service route carries one class; confirmation, retries and scopes rely on it."""

import subprocess
import sys
from pathlib import Path

import pytest

from histopilot.api import create_app
from histopilot.api.route_classes import (
    ADMIN,
    CLASSES,
    COMMIT,
    PREVIEW,
    READ,
    ROUTE_CLASSES,
    route_class,
)
from histopilot.config import Settings

API_SOURCES = Path(__file__).resolve().parents[1] / "histopilot" / "api"


@pytest.fixture(scope="module")
def routes(tmp_path_factory):
    root = tmp_path_factory.mktemp("route-classes")
    app = create_app(Settings(workspace=root / "workspace", static_dir=root))
    # The in-process schema lists every route; the service only hides the endpoint.
    return {
        (method.upper(), path)
        for path, operations in app.openapi()["paths"].items()
        if path.startswith("/api/")
        for method in operations
    }


def test_every_route_has_exactly_one_known_class(routes):
    missing = sorted(routes - ROUTE_CLASSES.keys())
    stale = sorted(ROUTE_CLASSES.keys() - routes)
    assert not missing, f"Classify these routes in histopilot/api/route_classes.py: {missing}"
    assert not stale, f"These classified routes no longer exist: {stale}"
    assert set(ROUTE_CLASSES.values()) <= set(CLASSES)


def test_no_route_hides_from_the_schema_the_check_reads():
    hidden = [
        path.name
        for path in API_SOURCES.glob("*.py")
        if "include_in_schema" in path.read_text(encoding="utf-8")
    ]
    assert not hidden, "A route outside the schema would escape classification"


def test_reads_never_change_state_and_previews_are_named_or_drafts():
    for (method, path), route_class_ in ROUTE_CLASSES.items():
        if method == "GET":
            assert route_class_ == READ, path
        if path.endswith("preview"):
            assert route_class_ == PREVIEW, path
    unnamed = {
        (method, path.removeprefix("/api/v1/projects/{identity}"))
        for (method, path), route_class_ in ROUTE_CLASSES.items()
        if route_class_ == PREVIEW and not path.endswith("preview")
    }
    # Draft saves are preview: an agent may prepare work that a person then freezes.
    assert unnamed == {
        ("POST", "/drafts"),
        ("PATCH", "/drafts/{draft_id}"),
        ("POST", "/model-experiments"),
        ("PATCH", "/model-experiments/{experiment_id}"),
        ("POST", "/model-experiments/{experiment_id}/setup-inputs"),
    }


@pytest.mark.parametrize(
    ("method", "path", "expected"),
    [
        ("GET", "/api/v1/projects/p1/model-experiments/headlines", READ),
        ("GET", "/api/v1/projects/p1/model-experiments/e1", READ),
        ("patch", "/api/v1/projects/p1/drafts/d1", PREVIEW),
        ("POST", "/api/v1/projects/p1/model-experiments/e1/submit", COMMIT),
        ("POST", "/api/v1/projects/p1/evaluation-runs/r1/scores", READ),
        ("POST", "/api/v1/projects/p1/operations/archives", ADMIN),
        ("POST", "/api/v1/task-center/owners/k/hold", COMMIT),
        ("PUT", "/api/v1/task-center/capacity", ADMIN),
        ("GET", "/api/v1/projects/p1/mil-experiments/batches/b/oof/c/1/2/slide.csv", READ),
        ("GET", "/api/v1/projects/p1/storage?refresh=1", READ),
        ("GET", "/api/v1/projects/", READ),
        ("DELETE", "/api/v1/projects/p1", None),
        ("GET", "/api/v1/unknown", None),
    ],
)
def test_concrete_requests_resolve_to_their_route_class(method, path, expected):
    assert route_class(method, path) == expected


def test_the_client_reads_the_tables_without_building_the_service():
    probe = (
        "import sys, histopilot.api.route_classes; "
        "loaded = [m for m in ('fastapi', 'histopilot.api.app') if m in sys.modules]; "
        "raise SystemExit(', '.join(loaded) or None)"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
