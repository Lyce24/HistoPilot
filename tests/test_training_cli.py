"""CLI training commands route the same explicit intents as the browser."""

import json

import pytest
from typer.testing import CliRunner

from histopilot.cli import app


@pytest.fixture
def api_calls(monkeypatch):
    calls = []

    def call(url, path, payload=None):
        calls.append((url, path, payload))
        return {"status": "queued", "batchId": "configuration-batch"}

    monkeypatch.setattr("histopilot.cli._feature_api", call)
    return calls


@pytest.mark.parametrize("resume", [False, True])
def test_train_batch_preserves_explicit_operation_and_escapes_ids(api_calls, resume):
    args = [
        "train-batch",
        "batch/one",
        "--project",
        "project/one",
        "--operation-id",
        "retry-once",
        "--url",
        "http://localhost:8899",
    ]
    if resume:
        args.append("--resume")
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert api_calls == [
        (
            "http://localhost:8899",
            "/projects/project%2Fone/mil-experiments/batches/batch%2Fone/"
            + ("resume" if resume else "launch"),
            {"operationId": "retry-once"},
        )
    ]
    assert json.loads(result.stdout)["status"] == "queued"


def test_train_batch_generates_distinct_operation_ids_when_not_supplied(api_calls):
    args = ["train-batch", "configuration-batch", "--project", "project-one"]
    for _ in range(2):
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output
    identities = [call[2]["operationId"] for call in api_calls]
    assert all(identity.startswith("training:") for identity in identities)
    assert len(set(identities)) == 2


@pytest.mark.parametrize(
    "option,suffix", [(None, "execution"), ("--results", "results"), ("--cancel", "cancel")]
)
def test_training_status_reads_results_or_requests_cancel(api_calls, option, suffix):
    args = ["training-status", "configuration-batch", "--project", "project-one"]
    if option:
        args.append(option)
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert (
        api_calls[0][1]
        == f"/projects/project-one/mil-experiments/batches/configuration-batch/{suffix}"
    )
    if suffix == "cancel":
        assert api_calls[0][2]["operationId"].startswith("cancel:")
    else:
        assert api_calls[0][2] is None


@pytest.mark.parametrize(
    "args",
    [
        ["training-status", "batch", "--project", "project", "--cancel", "--results"],
        ["train-batch", "batch"],
        ["training-status", "batch"],
        ["train-batch", "batch", "--project", "project", "--results"],
    ],
)
def test_invalid_training_cli_options_do_not_call_service(api_calls, args):
    result = CliRunner().invoke(app, args)
    assert result.exit_code != 0
    assert not api_calls


def test_older_commands_name_their_replacement_on_stderr_only(api_calls):
    result = CliRunner().invoke(app, ["train-batch", "configuration-batch", "--project", "p"])
    assert result.exit_code == 0
    assert "histopilot experiment start" in result.stderr
    assert json.loads(result.stdout)["status"] == "queued"


def test_a_batch_owned_by_an_experiment_explains_the_experiment_flow():
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Service(BaseHTTPRequestHandler):
        def reply(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802 - the http.server hook name
            self.reply(200, {"token": "t"})

        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers["Content-Length"]))
            refusal = "Start the experiment before training."
            self.reply(409, {"detail": refusal, "code": "EXPERIMENT_SUBMISSION_REQUIRED"})

        def log_message(self, *_arguments):
            return

    server = HTTPServer(("127.0.0.1", 0), Service)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        args = ["train-batch", "configuration-batch", "--project", "p", "--url", url]
        result = CliRunner().invoke(app, args)
    finally:
        server.shutdown()
    assert result.exit_code == 1
    assert "Start the experiment before training." in result.stderr
    assert "histopilot experiment freeze EXPERIMENT" in result.stderr
