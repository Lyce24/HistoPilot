"""One client for the local service, shared by the CLI and the agent tools."""

import json
import threading
import time
from contextlib import contextmanager
from urllib.parse import quote, urlencode

from histopilot.api.route_classes import (
    ADMIN,
    API_PREFIX,
    COMMIT,
    READ,
    TOKEN_PREFIX,
    route_class,
)

from .errors import ClientError, from_response
from .journal import OperationJournal, new_operation_id, operation_key
from .transport import (
    DEFAULT_URL,
    HttpTransport,
    Response,
    Transport,
    TransportError,
    TransportTimeout,
    service_url,
)

API = API_PREFIX
# The browser's schedule for reads that meet a busy project (web/src/api/client.ts).
BUSY_READ_RETRY_DELAYS = (0.25, 0.5, 1.0, 2.0)
# Previews, starts and results are synchronous and can take minutes on large studies.
REQUEST_TIMEOUT = 600.0


def segment(value: str) -> str:
    """One path segment, quoted so an ID can never change the route."""
    return quote(str(value), safe="")


class Client:
    def __init__(
        self,
        url: str = DEFAULT_URL,
        *,
        transport: Transport | None = None,
        timeout: float | None = None,
        host_header: str | None = None,
        journal: OperationJournal | None = None,
        token: str | None = None,
        sleep=time.sleep,
        clock=time.monotonic,
    ):
        self.url = service_url(url)
        # A token given here (such as a scoped agent token) is used as is: the client
        # never fetches the service's full-access session token instead.
        self.static_token = token
        self.transport = transport or HttpTransport(self.url, host_header=host_header)
        self.journal = journal
        # Journaled operations the service answered. They settle when the command ends
        # (`settle_answered`), so a command stopped while it waits replays them.
        self._answered: list[str] = []
        # The latest answer to a commit, and how many have come.
        self.answer_count, self.last_answer = 0, None
        self._clock, self._sleep = clock, sleep
        self.deadline = None if timeout is None else clock() + timeout
        # Limits of calls in progress, per thread: agent tools run side by side.
        self._limits = threading.local()
        self._session: dict | None = None

    # Time -----------------------------------------------------------------------------

    def remaining(self) -> float | None:
        """Seconds left before the nearest deadline: the client's own or this call's."""
        limits = [
            value
            for value in (self.deadline, getattr(self._limits, "deadline", None))
            if value is not None
        ]
        return min(limits) - self._clock() if limits else None

    @contextmanager
    def time_limit(self, seconds: float):
        """Bound the requests and pauses this thread makes inside to ``seconds``."""
        previous = getattr(self._limits, "deadline", None)
        limit = self._clock() + seconds
        self._limits.deadline = limit if previous is None else min(previous, limit)
        try:
            yield
        finally:
            self._limits.deadline = previous

    def pause(self, seconds: float) -> None:
        """Sleep, but never past the command's deadline."""
        remaining = self.remaining()
        if remaining is not None and remaining <= seconds:
            raise self.timeout_error()
        self._sleep(seconds)

    def timeout_error(self) -> ClientError:
        return ClientError(
            "The time limit ran out; any work already started goes on.",
            code="TIMEOUT",
            kind="timeout",
        )

    @property
    def scoped(self) -> bool:
        """True with an agent's scoped token: one project, never the machine."""
        return bool(self.static_token and self.static_token.startswith(TOKEN_PREFIX))

    # Session --------------------------------------------------------------------------

    def session(self) -> dict:
        """The service's session: its token and capabilities. The token changes on restart."""
        if self._session is None and self.static_token:
            self._session = {"token": self.static_token}
        if self._session is None:
            response = self._send("GET", f"{API}/session", headers={}, body=None)
            if (
                response.status == 401
                and from_response(response.status, response.body).code == "LOGIN_REQUIRED"
            ):
                secret = self._login_secret()
                if secret:
                    headers = {"X-HistoPilot-Login": secret}
                    response = self._send("GET", f"{API}/session", headers=headers, body=None)
            if response.status != 200:
                error = from_response(response.status, response.body)
                raise ClientError(
                    f"The service at {self.url} refused a session: {error.message}",
                    code="SESSION_REFUSED" if error.code.startswith("HTTP_") else error.code,
                    kind="unavailable",
                    status=response.status,
                )
            try:
                session = json.loads(response.body)
                session["token"].encode("ascii")
            except (ValueError, KeyError, TypeError, AttributeError, UnicodeError) as error:
                raise ClientError(
                    f"{self.url} does not look like a HistoPilot service.",
                    code="SESSION_REFUSED",
                    kind="unavailable",
                    status=response.status,
                ) from error
            self._session = session
        return self._session

    def _login_secret(self) -> str | None:
        """The sign-in secret: HISTOPILOT_LOGIN, else this user's file for the port."""
        import os
        from urllib.parse import urlsplit

        if os.environ.get("HISTOPILOT_LOGIN"):
            return os.environ["HISTOPILOT_LOGIN"]
        from histopilot.api.login import read_secret

        port = urlsplit(self.url).port or 80
        return read_secret(port)

    # Requests -------------------------------------------------------------------------

    def request(self, method: str, path: str, *, query=None, body=None, accept: str = "json"):
        """Send one request. ``path`` is relative to ``/api/v1`` unless it starts with /api/.
        ``accept`` is json, text, bytes, or response for the whole response with its headers.

        Reads that meet a busy project wait and read again, as the browser does. Nothing
        else is retried, apart from renewing the session token once.
        """
        method = method.upper()
        target = path if path.startswith("/api/") else API + path
        route = route_class(method, target) or COMMIT
        if query:
            pairs = [
                (key, _query_value(value)) for key, value in query.items() if value is not None
            ]
            if pairs:
                target += "?" + urlencode(pairs)
        payload = None if body is None else json.dumps(body).encode("utf-8")
        renewed, busy = False, 0
        while True:
            headers = {"X-HistoPilot-Token": self.session()["token"], "Accept": "*/*"}
            if payload is not None:
                headers["Content-Type"] = "application/json"
            response = self._send(method, target, headers=headers, body=payload)
            if response.status == 401 and not renewed and not self.static_token:
                self._session, renewed = None, True
                continue
            if response.status >= 400:
                error = from_response(response.status, response.body)
                if error.code == "PROJECT_BUSY" and route == READ:
                    if busy < len(BUSY_READ_RETRY_DELAYS):
                        self.pause(BUSY_READ_RETRY_DELAYS[busy])
                        busy += 1
                        continue
                raise error
            if response.status == 202:
                parked = _parked(response)
                if parked is not None:
                    raise parked
            answer = _decode(response, accept)
            if route in (COMMIT, ADMIN):
                self.answer_count += 1
                self.last_answer = answer
            return answer

    def get(self, path: str, **options):
        return self.request("GET", path, **options)

    def operation(
        self,
        method: str,
        path: str,
        body: dict,
        *,
        prefix: str,
        operation_id: str | None = None,
    ):
        """A commit that takes an operation ID. The ID is written to the journal before
        the request goes out and stays there until the command ends, so rerunning after a
        lost answer, a timeout or Ctrl-C replays the same operation.
        """
        if operation_id or self.journal is None:
            operation = operation_id or new_operation_id(prefix)
            return self.request(method, path, body={**body, "operationId": operation})
        key = operation_key(self.url, method, path, body)
        if key in self._answered:
            # Answered earlier in this command: sending it again is new work.
            self._answered.remove(key)
            self.journal.settle(key)
        operation = self.journal.begin(key, prefix, f"{method.upper()} {path}")
        try:
            result = self.request(method, path, body={**body, "operationId": operation})
        except ClientError as error:
            # The service answered and refused: a new attempt deserves a new ID. Keep it
            # when the outcome is unknown (timeouts, lost connections, server faults).
            if error.status is not None and error.status < 500:
                self.journal.settle(key)
            raise
        self._answered.append(key)
        return result

    def settle_answered(self) -> None:
        """Forget the IDs of the operations the service answered: the command is over, and
        running it again is new work."""
        if self.journal is not None:
            for key in self._answered:
                self.journal.settle(key)
        self._answered.clear()

    def _send(self, method, path, *, headers, body) -> Response:
        remaining = self.remaining()
        if remaining is not None and remaining <= 0:
            raise self.timeout_error()
        timeout = REQUEST_TIMEOUT if remaining is None else min(REQUEST_TIMEOUT, remaining)
        try:
            return self.transport.send(method, path, headers=headers, body=body, timeout=timeout)
        except TransportTimeout as error:
            raise ClientError(str(error), code="TIMEOUT", kind="timeout") from error
        except TransportError as error:
            raise ClientError(
                f"{str(error).rstrip('.')}. Is the service running? Start it with "
                "`histopilot serve`.",
                code="SERVICE_UNREACHABLE",
                kind="unavailable",
            ) from error


def _query_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list | tuple | set):
        return ",".join(str(item) for item in value)
    return str(value)


def _parked(response: Response) -> ClientError | None:
    """A scoped token's commit, parked by the service until a person approves it. Nothing
    changed yet, so it is reported as unconfirmed, with the request to approve."""
    try:
        body = json.loads(response.body)
    except ValueError:
        return None
    if not isinstance(body, dict) or body.get("code") != "CONFIRMATION_PENDING":
        return None
    return ClientError(
        body.get("detail") or "A person must approve this change.",
        code="CONFIRMATION_PENDING",
        kind="unconfirmed",
        status=202,
        data={"requestId": body.get("requestId"), "expiresAt": body.get("expiresAt")},
    )


def _decode(response: Response, accept: str):
    if accept == "response":
        return response
    if accept == "bytes":
        return response.body
    if accept == "text":
        return response.body.decode("utf-8", "replace")
    if not response.body:
        return None
    if "json" in response.content_type:
        try:
            return json.loads(response.body)
        except ValueError as error:
            raise ClientError(
                "The service's answer says it is JSON but cannot be read as JSON.",
                code="RESPONSE_UNREADABLE",
                kind="unavailable",
                status=response.status,
            ) from error
    if response.body[:1] in (b"{", b"["):
        try:
            return json.loads(response.body)
        except ValueError:
            pass  # text that only starts like JSON, such as a log line
    return response.body.decode("utf-8", "replace")
