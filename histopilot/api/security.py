"""Loopback browser boundary and a process-local session capability."""

from secrets import compare_digest

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from histopilot.api import login
from histopilot.api.responses import refusal as _refusal
from histopilot.api.route_classes import TOKEN_PREFIX
from histopilot.config import Settings


def configure_browser_boundary(
    app: FastAPI,
    settings: Settings,
    token: str,
    *,
    scoped=None,
    after_session=None,
    login_secret: str | None = None,
) -> None:
    """``scoped`` handles requests carrying an agent's `hpt_` token (histopilot/api/scopes.py);
    ``after_session`` sees every answered session-token request, for the audit log."""
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
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Content-Type", "X-HistoPilot-Token"],
    )

    cookie = login.cookie_name(settings.port)

    def current_secret() -> str | None:
        """The secret as its file holds it now, so `histopilot login rotate` signs everyone
        out at once; without the file, no one signs in."""
        return login.read_secret(settings.port) if login_secret else None

    def signed_in(request: Request) -> bool:
        secret = current_secret()
        if not secret:
            return False
        presented = request.headers.get(login.HEADER, "")
        if presented and compare_digest(presented.encode(), secret.encode()):
            return True
        value = request.cookies.get(cookie, "")
        # Bytes: a cookie the browser was handed can still hold any character.
        return bool(value) and compare_digest(value.encode(), login.cookie_value(secret).encode())

    def sign_in(request: Request):
        # Opening the printed link once leaves a cookie that only /session reads.
        response = RedirectResponse("/", status_code=303)
        offered = request.query_params.get("login", "")
        secret = current_secret()
        if secret and compare_digest(offered.encode(), secret.encode()):
            response.set_cookie(
                cookie,
                login.cookie_value(secret),
                httponly=True,
                samesite="strict",
                path="/api/v1/session",
            )
        return response

    @app.middleware("http")
    async def protect_local_service(request: Request, call_next):
        host_values = request.headers.getlist("host")
        origin_values = request.headers.getlist("origin")
        origin = request.headers.get("origin")
        fetch_site = request.headers.get("sec-fetch-site")
        needs_token = (
            request.url.path.startswith("/api/")
            and request.url.path not in {"/api/v1/health", "/api/v1/session"}
            and request.method != "OPTIONS"
        )
        supplied = request.headers.get("x-histopilot-token", "")
        if len(host_values) != 1 or host_values[0].lower() not in hosts:
            response = _refusal("Unrecognized local service Host.", "HOST_REFUSED", 400)
        elif len(origin_values) > 1 or (origin is not None and origin not in origins):
            response = _refusal("This browser Origin is not permitted.", "ORIGIN_REFUSED")
        elif fetch_site == "cross-site":
            response = _refusal(
                "Cross-site browser requests are not permitted.", "CROSS_SITE_REFUSED"
            )
        elif fetch_site not in {None, "same-origin", "same-site", "none"}:
            response = _refusal("Unrecognized browser request context.", "BROWSER_CONTEXT_REFUSED")
        elif fetch_site == "same-site" and origin is None:
            response = _refusal(
                "Same-site browser requests require an approved Origin.", "ORIGIN_REQUIRED"
            )
        elif (
            request.url.path == "/api/v1/session"
            and supplied.startswith(TOKEN_PREFIX)
            and scoped is not None
        ):
            # An agent's token asking for the full session is refused and audited there.
            response = await scoped(request, supplied, call_next)
        elif login_secret and request.url.path == "/" and "login" in request.query_params:
            response = sign_in(request)
        elif login_secret and request.url.path == "/api/v1/session" and not signed_in(request):
            response = _refusal(
                "This service requires sign-in: open the link it printed when it started, "
                "or run `histopilot login url` as the same user.",
                "LOGIN_REQUIRED",
                401,
            )
        elif needs_token and scoped is not None and supplied.startswith(TOKEN_PREFIX):
            response = await scoped(request, supplied, call_next)
        elif needs_token and not compare_digest(supplied.encode("utf-8"), token.encode("utf-8")):
            response = _refusal(
                "A valid local session token is required.", "SESSION_TOKEN_REQUIRED", 401
            )
        else:
            response = await call_next(request)
            if needs_token and after_session is not None:
                await after_session(request, response)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response
