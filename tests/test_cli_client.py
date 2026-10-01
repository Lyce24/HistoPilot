"""The client library: sessions, retries, error kinds, deadlines and the operation journal."""

import json

import pytest

from histopilot.client import Client, ClientError
from histopilot.client.errors import from_response
from histopilot.client.journal import OperationJournal
from histopilot.client.states import run_state, task_run_state
from histopilot.client.transport import (
    Response,
    TransportError,
    TransportTimeout,
    service_url,
)


def reply(payload, status=200):
    return Response(status, json.dumps(payload).encode(), {"content-type": "application/json"})


SESSION = reply({"token": "token-1"})


class Scripted:
    """A transport that answers from a script and records what was sent."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.sent = []

    def send(self, method, path, *, headers, body, timeout):
        self.sent.append(
            {"method": method, "path": path, "headers": headers, "body": body, "timeout": timeout}
        )
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def client(*answers, **options):
    transport = Scripted(*answers)
    pauses = []
    options.setdefault("sleep", pauses.append)
    return Client(transport=transport, **options), transport, pauses


def test_the_session_token_is_sent_and_renewed_once_after_a_restart():
    service, transport, _ = client(
        SESSION,
        reply({"detail": "A valid local session token is required."}, 401),
        reply({"token": "token-2"}),
        reply({"projects": []}),
    )
    assert service.get("/projects") == {"projects": []}
    paths = [request["path"] for request in transport.sent]
    assert paths == ["/api/v1/session", "/api/v1/projects", "/api/v1/session", "/api/v1/projects"]
    assert transport.sent[-1]["headers"]["X-HistoPilot-Token"] == "token-2"


def test_a_second_refusal_after_renewal_is_a_session_problem():
    refused = reply({"detail": "A valid local session token is required."}, 401)
    service, _, _ = client(SESSION, refused, SESSION, refused)
    with pytest.raises(ClientError) as raised:
        service.get("/projects")
    assert (raised.value.code, raised.value.kind, raised.value.exit_code) == (
        "SESSION_REFUSED",
        "unavailable",
        6,
    )


def test_busy_reads_retry_on_the_browsers_schedule_and_commits_never_do():
    busy = reply(
        {"detail": "Another operation is changing this workspace.", "code": "PROJECT_BUSY"}, 409
    )
    service, _, pauses = client(SESSION, busy, busy, reply({"items": []}))
    assert service.get("/projects/p/model-experiments") == {"items": []}
    assert pauses == [0.25, 0.5]

    service, _, pauses = client(SESSION, *[busy] * 5)
    with pytest.raises(ClientError) as raised:
        service.get("/projects/p/model-experiments")
    assert raised.value.exit_code == 4 and pauses == [0.25, 0.5, 1.0, 2.0]

    service, transport, pauses = client(SESSION, busy)
    with pytest.raises(ClientError):
        service.request("POST", "/projects/p/model-experiments/e/submit", body={})
    assert pauses == [] and len(transport.sent) == 2


@pytest.mark.parametrize(
    ("status", "body", "code", "kind", "exit_code"),
    [
        (
            404,
            {"detail": "No such experiment.", "code": "EXPERIMENT_NOT_FOUND"},
            "EXPERIMENT_NOT_FOUND",
            "not-found",
            5,
        ),
        (
            409,
            {"detail": "Preview again.", "code": "PREVIEW_STALE"},
            "PREVIEW_STALE",
            "conflict",
            4,
        ),
        (
            409,
            {"detail": "Reused.", "code": "OPERATION_CONFLICT"},
            "OPERATION_CONFLICT",
            "refused",
            3,
        ),
        (
            422,
            {"detail": "Bad split.", "code": "INVALID_STRATEGY_CONFIG"},
            "INVALID_STRATEGY_CONFIG",
            "invalid",
            2,
        ),
        (400, {"detail": "Unrecognized local service Host."}, "SESSION_REFUSED", "unavailable", 6),
        (500, "Internal Server Error", "HTTP_500", "internal", 1),
    ],
)
def test_service_errors_map_to_kinds_and_exit_codes(status, body, code, kind, exit_code):
    raw = body.encode() if isinstance(body, str) else json.dumps(body).encode()
    error = from_response(status, raw)
    assert (error.code, error.kind, error.exit_code, error.status) == (
        code,
        kind,
        exit_code,
        status,
    )


def test_request_validation_becomes_one_finding_per_field():
    body = {
        "detail": [
            {
                "type": "extra_forbidden",
                "loc": ["body", "splitUnit2"],
                "msg": "Extra inputs are not permitted",
            },
            {"type": "missing", "loc": ["body", "target", "field"], "msg": "Field required"},
        ]
    }
    error = from_response(422, json.dumps(body).encode())
    assert (error.code, error.kind, error.exit_code) == ("REQUEST_INVALID", "invalid", 2)
    assert [finding["field"] for finding in error.findings] == ["splitUnit2", "target.field"]
    assert "splitUnit2: Extra inputs are not permitted" in error.message


def test_an_unreachable_service_and_a_silent_one_are_told_apart():
    service, _, _ = client(TransportError("Cannot reach http://127.0.0.1:8787: refused"))
    with pytest.raises(ClientError) as raised:
        service.get("/projects")
    assert (raised.value.code, raised.value.exit_code) == ("SERVICE_UNREACHABLE", 6)
    assert "histopilot serve" in raised.value.message

    service, _, _ = client(SESSION, TransportTimeout("No answer"))
    with pytest.raises(ClientError) as raised:
        service.get("/projects")
    assert (raised.value.code, raised.value.exit_code) == ("TIMEOUT", 8)


def test_the_deadline_bounds_every_request_and_every_pause():
    now = [100.0]
    service, transport, _ = client(
        SESSION, reply({"projects": []}), timeout=30, clock=lambda: now[0]
    )
    service.get("/projects")
    assert transport.sent[-1]["timeout"] == 30
    now[0] = 131.0
    with pytest.raises(ClientError) as raised:
        service.get("/projects")
    assert raised.value.kind == "timeout" and len(transport.sent) == 2


def test_the_journal_replays_an_operation_until_the_service_answers(tmp_path):
    journal = OperationJournal(tmp_path / "operations")
    path, body = "/projects/p/model-experiments/e/submit", {"expectedRevision": 3}
    service, transport, _ = client(
        SESSION,
        TransportError("connection reset"),
        reply({"status": "submitted"}, 202),
        reply({"status": "submitted"}, 202),
        journal=journal,
    )
    with pytest.raises(ClientError):
        service.operation("POST", path, body, prefix="submit")
    first = json.loads(transport.sent[-1]["body"])["operationId"]
    service.operation("POST", path, body, prefix="submit")
    assert json.loads(transport.sent[-1]["body"])["operationId"] == first
    service.operation("POST", path, body, prefix="submit")
    assert json.loads(transport.sent[-1]["body"])["operationId"] != first
    assert first.startswith("submit:")
    service.settle_answered()
    assert list((tmp_path / "operations").iterdir()) == []


def test_an_answered_operation_replays_until_its_command_ends(tmp_path):
    # A command stopped while it waits for the work it started (a timeout, Ctrl-C) leaves
    # its operation journaled: the rerun replays it instead of starting the work again.
    journal = OperationJournal(tmp_path / "operations")
    path, body = "/projects/p/model-experiments/e/submit", {"expectedRevision": 3}
    first, transport, _ = client(SESSION, reply({"status": "submitted"}, 202), journal=journal)
    assert first.operation("POST", path, body, prefix="submit") == {"status": "submitted"}
    assert (first.answer_count, first.last_answer) == (1, {"status": "submitted"})
    sent = json.loads(transport.sent[-1]["body"])["operationId"]
    rerun, transport, _ = client(SESSION, reply({"status": "submitted"}, 202), journal=journal)
    rerun.operation("POST", path, body, prefix="submit")
    assert json.loads(transport.sent[-1]["body"])["operationId"] == sent
    rerun.settle_answered()
    later, transport, _ = client(SESSION, reply({"status": "submitted"}, 202), journal=journal)
    later.operation("POST", path, body, prefix="submit")
    assert json.loads(transport.sent[-1]["body"])["operationId"] != sent


def test_a_refusal_settles_the_operation_and_a_server_fault_keeps_it(tmp_path):
    journal = OperationJournal(tmp_path / "operations")
    path = "/projects/p/model-experiments/e/submit"
    service, transport, _ = client(
        SESSION,
        reply({"detail": "Stale.", "code": "REVISION_CONFLICT"}, 409),
        reply({"detail": "Boom."}, 500),
        reply({}, 202),
        journal=journal,
    )
    for _ in range(2):
        with pytest.raises(ClientError):
            service.operation("POST", path, {"expectedRevision": 1}, prefix="submit")
    ids = [json.loads(request["body"])["operationId"] for request in transport.sent[1:]]
    service.operation("POST", path, {"expectedRevision": 1}, prefix="submit")
    last = json.loads(transport.sent[-1]["body"])["operationId"]
    assert ids[0] != ids[1] and last == ids[1]


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:8787",
        "http://example.org:8787",
        "http://user:pw@127.0.0.1:8787",
        "http://127.0.0.1:8787/api",
    ],
)
def test_only_plain_loopback_service_urls_are_accepted(url):
    with pytest.raises(ValueError):
        service_url(url)
    assert service_url("http://localhost:8799/") == "http://localhost:8799"


def test_run_states_unify_the_service_vocabularies():
    assert [
        run_state(value) for value in ("blocked", "running", "completed", "held", "interrupted")
    ] == [
        "queued",
        "running",
        "succeeded",
        "needs-attention",
        "needs-attention",
    ]
    assert run_state("planned") is None and run_state(None) is None
    # Runs, interpretations and Apply batches spell it with an underscore.
    assert run_state("not_started") is None and run_state("not-started") is None
    assert task_run_state({"state": "interrupted", "awaitingRequeue": True}) == "queued"
    # A held owner's queued task never starts on its own, so a wait ends on it.
    assert task_run_state({"state": "queued", "owner": {"held": True}}) == "needs-attention"
    assert task_run_state({"state": "queued", "owner": {"held": False}}) == "queued"
    assert task_run_state({"state": "failed"}) == "failed"


def test_a_given_token_is_used_as_is_and_never_swapped_for_the_session_token():
    refused = reply({"detail": "A valid local session token is required."}, 401)
    service, transport, _ = client(reply({"projects": []}), refused, token="hpt_scoped")
    assert service.get("/projects") == {"projects": []}
    with pytest.raises(ClientError) as raised:
        service.get("/projects")
    assert raised.value.kind == "unavailable"
    assert [request["path"] for request in transport.sent] == ["/api/v1/projects"] * 2
    assert {request["headers"]["X-HistoPilot-Token"] for request in transport.sent} == {
        "hpt_scoped"
    }
