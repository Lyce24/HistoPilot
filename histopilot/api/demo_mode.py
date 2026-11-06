"""The public demo's boundary: what `histopilot serve --demo` answers, and to whom.

A demo service serves the browser UI and the packaged synthetic BLCA demo to anyone who
reaches it, usually through a reverse proxy such as a Hugging Face Space's. It trusts nothing
about a request: not its client address, Host, Origin, fetch metadata or forwarded headers,
none of which it reads. It needs to trust nothing, because of what it answers:

- GET, HEAD and OPTIONS only. Every other method is refused before any route runs.
- Under /api/, only the reads in DEMO_ROUTES, each the same public, synthetic content for
  every caller. Every other API route is refused for every method, including the reads that
  would describe this machine: folders and data roots, the workspace and its paths, the
  Task Center, tokens and agent requests, slide images and exports.
- Outside /api/, the packaged UI's files, which the static route confines to its folder.

The listed reads still require the session token, so the UI runs unchanged, but the token is
no secret here: anyone can fetch it, and it reaches nothing more. A service started without
--demo never installs this boundary; it keeps the loopback one in security.py.

A demo can also serve packaged projects (`--demo-projects`): real studies copied into the
throwaway workspace when the service starts, shown through the full application. For those
projects alone it answers exactly the routes route_classes.py classes as reads, whatever their
method, except the few in PROJECT_REFUSED, which reach past the project's own records. Every
other route of theirs is refused, as is every route of any other project.
"""

from secrets import compare_digest

from fastapi import FastAPI, Request

from histopilot.application.blca_demo import DEMO_ID

from .responses import refusal
from .route_classes import API_PREFIX, READ, expand, route_class, route_template

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

# Every API path a demo service answers, exactly as the router matches it, and why it is safe.
DEMO_ROUTES = {
    "/api/v1/health": "Status and version, the same for every caller.",
    "/api/v1/session": "The page's token, which in demo mode reaches only these reads.",
    "/api/v1/projects": "The start page's list: the demo alone, without a storage path.",
    f"/api/v1/projects/{DEMO_ID}/workspace": "The packaged synthetic fixture.",
}
# Answered without the token, as in the loopback boundary.
TOKEN_FREE = frozenset({"/api/v1/health", "/api/v1/session"})
# A Space's page on huggingface.co shows the demo in a frame; no other site may frame it.
FRAME_ANCESTORS = "frame-ancestors 'self' https://huggingface.co"
# Reads of a packaged project that a demo still refuses, and why: each reaches past the
# project's own records, to this machine or to a process started per request.
PROJECT_REFUSED = {
    expand(path): reason
    for path, reason in {
        "{P}/imports/inspect": "Inspects a file path on this machine.",
        "{P}/mil-experiments/runtime": "Starts the training interpreter to probe it.",
        "{P}/interpretations/gallery/thumbnail": "Renders slide images in a reader process.",
        "{P}/interpretations/{interpretation_id}/slides/{slide_id}/thumbnail": "Slide images.",
        "{P}/interpretations/{interpretation_id}/slides/{slide_id}/region": "Slide images.",
        "{P}/interpretations/{interpretation_id}/slides/{slide_id}/patches/{patch_index}/image": (
            "Slide images."
        ),
        "{P}/morphology/image": "Slide images.",
        "{P}/morphology/patch": "Slide images.",
        "{P}/morphology/patch-region": "Slide images.",
        "{P}/morphology/quality": "Reads slide images and features.",
        "{P}/morphology/slides": "Reads slide images and features.",
        "{P}/morphology/index": "Builds an index from patch features on this machine.",
        "{P}/morphology/neighbors": "Searches patch features on this machine.",
    }.items()
}
PROJECTS_PREFIX = f"{API_PREFIX}/projects/"


def packaged_project(path: str, projects: frozenset[str]) -> str | None:
    """The packaged demo project a request path is under, if any."""
    if not path.startswith(PROJECTS_PREFIX):
        return None
    identity = path[len(PROJECTS_PREFIX) :].split("/", 1)[0]
    return identity if identity in projects else None


def demo_refusal(method: str, path: str, projects: frozenset[str] = frozenset()):
    """The refusal for a request a demo service never answers; None for one it does."""
    if packaged_project(path, projects):
        # HEAD and OPTIONS ask about the GET route; the router answers them as it would.
        lookup = "GET" if method in {"HEAD", "OPTIONS"} else method
        kind = route_class(lookup, path)
        if kind == READ and route_template(lookup, path) not in PROJECT_REFUSED:
            return None
        if kind == READ or method in SAFE_METHODS:
            return refusal(
                "This public demo does not serve this request.", "DEMO_MODE_ROUTE_UNAVAILABLE"
            )
        return refusal("This public demo is read-only.", "DEMO_MODE_READ_ONLY")
    if method not in SAFE_METHODS:
        return refusal("This public demo is read-only.", "DEMO_MODE_READ_ONLY")
    if (path == "/api" or path.startswith("/api/")) and path not in DEMO_ROUTES:
        return refusal(
            "This public demo does not serve this route.",
            "DEMO_MODE_ROUTE_UNAVAILABLE",
        )
    return None


def configure_demo_boundary(
    app: FastAPI, token: str, projects: frozenset[str] = frozenset()
) -> None:
    """Install the demo boundary in place of the loopback one, CORS included: no other
    origin may read the demo's API from a browser. `projects` are the packaged projects'
    identities."""

    @app.middleware("http")
    async def protect_public_demo(request: Request, call_next):
        # The path the router matches; Request.url would end it at an encoded "?".
        path = request.scope["path"]
        supplied = request.headers.get("x-histopilot-token", "")
        response = demo_refusal(request.method, path, projects)
        if response is None and (
            (path in DEMO_ROUTES or packaged_project(path, projects))
            and path not in TOKEN_FREE
            and request.method != "OPTIONS"
            and not compare_digest(supplied.encode("utf-8"), token.encode("utf-8"))
        ):
            response = refusal(
                "A valid local session token is required.", "SESSION_TOKEN_REQUIRED", 401
            )
        if response is None:
            response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = FRAME_ANCESTORS
        if path == "/api" or path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response
