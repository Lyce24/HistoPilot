"""Loopback browser boundary and a process-local session capability."""

from secrets import compare_digest

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from histopilot.config import Settings


def configure_browser_boundary(app: FastAPI, settings: Settings, token: str) -> None:
    hosts = {f"{host}:{settings.port}" for host in ("127.0.0.1", "localhost", "[::1]")}
    if settings.port == 80:
        hosts.update({"127.0.0.1", "localhost", "[::1]"})
    origins = {f"http://{host}" for host in hosts}
    if settings.dev:
        origins.update({"http://localhost:5173", "http://127.0.0.1:5173", "http://[::1]:5173"})
    app.add_middleware(
        CORSMiddleware,
        allow_origins=sorted(origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Content-Type", "X-HistoPilot-Token"],
    )

    @app.middleware("http")
    async def protect_local_service(request: Request, call_next):
        reason = None
        status = 403
        host_values = request.headers.getlist("host")
        origin_values = request.headers.getlist("origin")
        origin = request.headers.get("origin")
        fetch_site = request.headers.get("sec-fetch-site")
        if len(host_values) != 1 or host_values[0].lower() not in hosts:
            reason, status = "Unrecognized local service Host.", 400
        elif len(origin_values) > 1 or (origin is not None and origin not in origins):
            reason = "This browser Origin is not permitted."
        elif fetch_site == "cross-site":
            reason = "Cross-site browser requests are not permitted."
        elif fetch_site not in {None, "same-origin", "same-site", "none"}:
            reason = "Unrecognized browser request context."
        elif fetch_site == "same-site" and origin is None:
            reason = "Same-site browser requests require an approved Origin."
        elif (
            request.url.path.startswith("/api/")
            and request.url.path
            not in {
                "/api/v1/health",
                "/api/v1/session",
            }
            and request.method != "OPTIONS"
        ):
            supplied = request.headers.get("x-histopilot-token", "")
            if not compare_digest(supplied.encode("utf-8"), token.encode("utf-8")):
                reason, status = "A valid local session token is required.", 401
        if reason:
            response = JSONResponse({"detail": reason}, status_code=status)
        else:
            response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response
