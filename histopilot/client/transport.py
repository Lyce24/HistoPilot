"""How the client reaches the local service: plain HTTP, replaceable for in-process tests."""

import http.client
from dataclasses import dataclass, field
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

DEFAULT_URL = "http://127.0.0.1:8787"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


@dataclass(frozen=True)
class Response:
    status: int
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)  # lower-case names

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "")


class TransportError(Exception):
    """The service could not be reached, or it stopped answering mid-request."""


class TransportTimeout(TransportError):
    """No answer before the deadline; a commit may or may not have been applied."""


class Transport(Protocol):
    def send(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> Response: ...


def service_url(url: str) -> str:
    """A loopback service URL without a trailing slash; anything else is refused."""
    parsed = urlsplit(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in LOOPBACK_HOSTS
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError(
            "Use an HTTP loopback service URL without credentials or a path, "
            f"such as {DEFAULT_URL}; reach a remote machine through an SSH forward."
        )
    return url.rstrip("/")


class HttpTransport:
    """Standard-library HTTP. An SSH forward to another local port needs ``host_header``:
    the service accepts only its own ``127.0.0.1:<port>`` as the Host."""

    def __init__(self, url: str, *, host_header: str | None = None):
        self.base = service_url(url)
        self.host_header = host_header
        # Loopback only: an HTTP proxy from the environment must never see the token.
        self._opener = build_opener(ProxyHandler({}))

    def send(self, method, path, *, headers, body, timeout):
        request = Request(self.base + path, data=body, headers=headers, method=method)
        if self.host_header:
            request.add_unredirected_header("Host", self.host_header)
        try:
            try:
                with self._opener.open(request, timeout=timeout) as response:
                    return Response(response.status, response.read(), _headers(response.headers))
            except HTTPError as error:
                # Reading the error's body can fail too; the handlers below report that.
                with error:
                    return Response(error.code, error.read(), _headers(error.headers))
        except TimeoutError as error:
            raise TransportTimeout(f"No answer from {self.base} in {timeout:g} s.") from error
        except URLError as error:
            if isinstance(error.reason, TimeoutError):
                raise TransportTimeout(f"No answer from {self.base} in {timeout:g} s.") from error
            raise TransportError(f"Cannot reach {self.base}: {error.reason}") from error
        except (OSError, http.client.HTTPException) as error:
            raise TransportError(f"Lost the connection to {self.base}: {error}") from error


def _headers(message) -> dict[str, str]:
    return {name.lower(): value for name, value in (message.items() if message else [])}
