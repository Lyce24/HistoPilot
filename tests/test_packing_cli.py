"""Feature CLI commands submit the same validated intent as the browser."""

import pytest
import typer
from typer.testing import CliRunner

from histopilot.cli import _feature_api, app


@pytest.fixture
def api_calls(monkeypatch):
    calls = []

    def call(url, path, payload=None):
        calls.append((url, path, payload))
        if path.endswith("/preview"):
            return {"canRun": True, "previewHash": "a" * 64, "spec": payload}
        return {"id": "packing-job", "state": "running"}

    monkeypatch.setattr("histopilot.cli._feature_api", call)
    return calls


def test_preview_is_read_only_and_default_submit_preserves_precision(api_calls):
    runner = CliRunner()
    args = ["pack-features", "configuration-source", "--project", "project-1"]
    preview = runner.invoke(app, [*args, "--preview"])
    assert preview.exit_code == 0, preview.output
    assert len(api_calls) == 1
    assert api_calls[0][2]["dtype"] == "preserve"
    api_calls.clear()
    started = runner.invoke(app, [*args, "--operation-id", "retry-safe"])
    assert started.exit_code == 0, started.output
    assert len(api_calls) == 2
    assert api_calls[-1][2]["previewHash"] == "a" * 64
    assert api_calls[-1][2]["operationId"] == "retry-safe"
    assert api_calls[-1][2]["featureSetId"] == "configuration-source"


def test_validation_only_uses_a_job_and_bad_precision_never_calls_service(api_calls):
    args = ["pack-features", "configuration-source", "--project", "project-1"]
    result = CliRunner().invoke(app, [*args, "--validate-only"])
    assert result.exit_code == 0, result.output
    assert api_calls[-1][2]["action"] == "validate"
    assert api_calls[-1][2]["outputPath"] is None
    api_calls.clear()
    result = CliRunner().invoke(app, [*args, "--dtype", "int8"])
    assert result.exit_code != 0
    assert not api_calls


def test_jobs_inspection_and_cancellation_use_explicit_job(api_calls):
    runner = CliRunner()
    result = runner.invoke(app, ["feature-jobs", "--project", "project/1"])
    assert result.exit_code == 0, result.output
    assert api_calls[-1][1] == "/projects/project%2F1/feature-packs"
    result = runner.invoke(
        app, ["feature-jobs", "--project", "p", "--job", "packing-a", "--cancel"]
    )
    assert result.exit_code == 0, result.output
    assert api_calls[-1][1].endswith("/packing-a/cancel")
    assert api_calls[-1][2] == {}
    result = runner.invoke(app, ["feature-jobs", "--project", "p", "--cancel"])
    assert result.exit_code != 0


def test_existing_pack_is_verified_through_the_same_worker(api_calls):
    args = [
        "pack-features",
        "configuration-source",
        "--project",
        "project-1",
        "--existing-pack",
        "/data/existing-pack",
    ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert api_calls[-1][2]["action"] == "attach"
    assert api_calls[-1][2]["existingPath"] == "/data/existing-pack"
    assert api_calls[-1][2]["outputPath"] is None
    for extra in (["--validate-only"], ["--dtype", "float16"], ["--output", "/data/new-pack"]):
        api_calls.clear()
        result = CliRunner().invoke(app, [*args, *extra])
        assert result.exit_code != 0
        assert not api_calls


@pytest.mark.parametrize(
    "url",
    ["https://example.org", "http://127.0.0.1/a", "http://user@localhost", "http://localhost?x=1"],
)
def test_feature_api_cannot_send_local_session_to_external_address(monkeypatch, url):
    monkeypatch.setattr(
        "histopilot.cli.urlopen", lambda *_a, **_k: pytest.fail("Must reject before network access")
    )
    with pytest.raises(typer.BadParameter):
        _feature_api(url, "/projects/p/feature-packs")
