"""Request IDs and access lines.

Every request gets an ID: a well-formed ``X-Request-ID`` from the client, else a new one.
It is echoed on the response, stamps every record logged while the request is served
(and the tasks it queues), and closes the request with one access line: the method, the
route template, the status, the duration and the kind of credential presented. Never a
path's values, the query, headers, cookies or the body.
"""

import logging
import time

from histopilot.api.route_classes import TOKEN_PREFIX, route_template
from histopilot.observability import logs

ACCESS = logging.getLogger("histopilot.api.access")
HEADER = b"x-request-id"


def route_of(scope) -> str:
    """The template of the route that served the request, never its concrete path."""
    path = getattr(scope.get("route"), "path", None)
    if isinstance(path, str):
        return path
    # Refused before routing (Host, Origin, session or scope checks): the route table
    # still names an API route.
    return route_template(scope.get("method", ""), scope.get("path", "")) or "unmatched"


def credential(headers) -> str:
    """The kind of credential a request presented, never the credential itself."""
    for name, value in headers:
        if name == b"x-histopilot-token" and value:
            return "token" if value.startswith(TOKEN_PREFIX.encode()) else "session"
    return "none"


class RequestLog:
    """ASGI middleware: the request's ID and its access line."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = scope.get("headers") or []
        offered = [value for name, value in headers if name == HEADER]
        identity = None
        if len(offered) == 1:
            identity = logs.valid_request_id(offered[0].decode("latin-1"))
        identity = identity or logs.new_request_id()
        # Left set for the rest of the request's task: when an error escapes the app, the
        # server's own report of it is made there and carries the ID too.
        logs.REQUEST_ID.set(identity)
        echoed = (HEADER, identity.encode("ascii"))
        status = 500
        started = time.perf_counter()

        async def answer(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message = {**message, "headers": [*message.get("headers", ()), echoed]}
            await send(message)

        try:
            await self.app(scope, receive, answer)
        finally:
            seconds = time.perf_counter() - started
            method, route = scope.get("method", ""), route_of(scope)
            ACCESS.info(
                "%s %s %s",
                method,
                route,
                status,
                extra={
                    "method": method,
                    "route": route,
                    "status": status,
                    "duration_ms": round(seconds * 1000, 1),
                    "auth": credential(headers),
                },
            )
