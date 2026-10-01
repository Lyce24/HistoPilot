"""`histopilot run …` sends the read routes' bodies and renders their answers."""

import json

import pytest
from support.cli import Service
from typer.testing import CliRunner

from histopilot.cli import app
from histopilot.client.transport import Response

RUN = "configuration-" + "a" * 64
OTHER = "configuration-" + "b" * 64
REFERENCE = "configuration-" + "c" * 64


class Recorder:
    """Answers every request from a table keyed by path suffix; records what was sent."""

    def __init__(self, answers):
        self.answers = answers
        self.sent = []

    def send(self, method, path, *, headers, body, timeout):
        if path.endswith("/session"):
            return Response(200, json.dumps({"token": "t"}).encode(), {})
        self.sent.append((method, path, json.loads(body) if body else None))
        for suffix, (content_type, payload) in self.answers.items():
            if path.split("?")[0].endswith(suffix):
                content = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                return Response(200, content, {"content-type": content_type})
        return Response(404, json.dumps({"detail": "No.", "code": "NOPE"}).encode(), {})


@pytest.fixture
def recorder(monkeypatch):
    from histopilot.commands import common

    answers = {
        "/scores": ("application/json", {"auroc": 0.91, "accuracy": 0.8, "confusion": [[1, 0]]}),
        "/agreement": ("application/json", {"sources": [], "pairs": []}),
        "/recalibration": ("application/json", {"unit": "slide", "before": {}, "after": {}}),
        "/performance/breakdown": ("application/json", {"groups": [], "overall": {}}),
        "/cases/query": (
            "application/json",
            {"items": [{"id": "s1"}], "total": 3, "offset": 0, "hasMore": True},
        ),
        "/cases/export": ("text/csv", b"slide,probability\ns1,0.9\n"),
        "/inference/summary": ("application/json", {"counts": {}}),
        "/inference/export": ("text/csv", b"slide,class\ns1,high\n"),
        "/evaluation-runs/compare": ("application/json", {"paired": True}),
        # Without --reference, the run is read to find its labels: a labeled run needs no other.
        f"/evaluation-runs/{RUN}": (
            "application/json",
            {"id": RUN, "manifest": {"purpose": "evaluation", "cohortId": "cohort-1"}},
        ),
    }
    transport = Recorder(answers)
    monkeypatch.setattr(common, "TRANSPORT_FACTORY", lambda _url: transport)
    return transport


def cli(*arguments):
    result = CliRunner().invoke(app, [*arguments, "--project", "project-1", "--json"])
    return result.exit_code, json.loads(result.stdout)


def test_each_read_sends_its_body(recorder):
    assert cli("run", "metrics", RUN, "--reference", REFERENCE)[1]["data"]["auroc"] == 0.91
    cli("run", "agreement", RUN, "--unit", "slide")
    cli("run", "recalibration", RUN, "--unit", "patient")
    cli("run", "subgroups", RUN, "--attribute", "site")
    cli("run", "summary", RUN, "--attribute", "site", "--comparison", OTHER)
    cli("run", "compare", RUN, OTHER)
    bodies = {path.rsplit("/", 1)[-1]: body for _, path, body in recorder.sent}
    assert bodies["scores"] == {"referenceId": REFERENCE}
    assert bodies["agreement"] == {"unit": "slide"}
    assert bodies["recalibration"] == {"unit": "patient"}
    assert bodies["breakdown"] == {"unit": "selected", "attribute": "site"}
    assert bodies["summary"] == {"unit": "selected", "attribute": "site", "comparisonId": OTHER}
    assert bodies["compare"] == {"leftEvaluationId": RUN, "rightEvaluationId": OTHER}
    reads = [path.split("?")[0] for method, path, _ in recorder.sent if method == "GET"]
    # Besides the run, only `run summary` reads more: the experiments and predictors that
    # name the model behind each run (see test_summaries_name_the_model_of_each_run).
    described = ("/model-experiments", "/predictors")
    assert all(
        path.endswith(f"/evaluation-runs/{RUN}") or path.endswith(described) for path in reads
    )
    assert len([method for method, _, _ in recorder.sent if method == "POST"]) == 6


def test_cases_are_paged_within_the_services_limit(recorder):
    code, envelope = cli("run", "cases", RUN, "--outcome", "error", "--limit", "500")
    assert code == 0 and envelope["data"] == [{"id": "s1"}]
    assert envelope["page"] == {"offset": 0, "limit": 100, "hasMore": True}
    body = recorder.sent[-1][2]
    assert body["outcome"] == "error" and body["limit"] == 100 and "memberDisagreement" not in body
    assert cli("run", "cases", RUN, "--outcome", "maybe")[0] == 2


def test_exports_are_written_and_never_overwrite_by_accident(recorder, tmp_path):
    target = tmp_path / "predictions.csv"
    code, envelope = cli("run", "export-predictions", RUN, "-o", str(target), "--attributes", "")
    assert code == 0 and target.read_bytes() == b"slide,class\ns1,high\n"
    assert recorder.sent[-1][2] == {"unit": "selected", "attributes": []}
    assert cli("run", "export-predictions", RUN, "-o", str(target))[0] == 2
    code, _ = cli("run", "export-cases", RUN, "-o", str(tmp_path / "cases.csv"))
    assert code == 0 and (tmp_path / "cases.csv").read_text().startswith("slide,probability")


def test_a_missing_run_is_not_found_in_the_real_service(tmp_path, monkeypatch):
    with Service(tmp_path, monkeypatch) as service:
        project = service.create_project()
        missing = service.cli("run", "metrics", RUN, "--project", project, "--json")
    assert missing.code == 5 and missing.envelope["error"]["status"] == 404


def test_summaries_name_the_model_of_each_run(recorder):
    # Predictor names omit their batch: the summary says which model made each run.
    def predictor(identity, batch, seed):
        return {
            "id": identity,
            "lifecycleState": "active",
            "manifest": {
                "experimentId": "draft-1",
                "batchId": batch,
                "candidateId": "c1",
                "candidateNumber": 1,
                "method": "ensemble",
                "trainingSeed": seed,
                "splitSeed": seed,
                "recipe": {"model": "abmil" if batch == "b-1" else "nnmil"},
            },
        }

    experiments = [
        {"id": "draft-1", "name": "Study", "batches": [{"id": "b-1", "name": "Baseline"}]},
    ]
    recorder.answers["/inference/summary"] = (
        "application/json",
        {
            "predictorId": "p-1",
            "count": 2,
            "classOrder": ["high", "low"],
            "predicted": [{"label": "high", "count": 1}, {"label": "low", "count": 1}],
            "comparison": {"predictorId": "p-2", "agreement": 0.5, "kappa": 0.0, "count": 2},
        },
    )
    recorder.answers["/model-experiments"] = ("application/json", {"items": experiments})
    recorder.answers["/predictors"] = (
        "application/json",
        {"items": [predictor("p-1", "b-1", 42), predictor("p-2", "b-2", 43)]},
    )
    code, envelope = cli("run", "summary", RUN, "--comparison", OTHER)
    assert code == 0, envelope
    data = envelope["data"]
    assert data["model"]["description"] == (
        "Baseline · Configuration 1 · Fold ensemble · Train 42 / split 42"
    )
    assert data["model"]["architecture"] == "abmil" and data["model"]["experiment"] == "Study"
    # A batch no experiment lists keeps a short form of its ID.
    assert data["comparison"]["model"]["description"].startswith("Batch b-2 · Configuration 1")


def test_cases_beside_another_runs_calls(recorder):
    recorder.answers["/cases/query"] = (
        "application/json",
        {
            "items": [
                {
                    "id": "s1",
                    "predictedLabel": "high",
                    "comparison": {"predictedLabel": "low", "disagrees": True},
                }
            ],
            "total": 1,
            "hasMore": False,
        },
    )
    result = CliRunner().invoke(
        app,
        ["run", "cases", RUN, "--comparison", OTHER, "--outcome", "disagreement"]
        + ["--project", "project-1"],
    )
    assert result.exit_code == 0, result.stdout
    header, row = result.stdout.splitlines()[:2]
    assert header.endswith("OTHER RUN") and row.rstrip().endswith("low")
    body = next(body for _, path, body in recorder.sent if path.endswith("/cases/query"))
    assert body["comparisonId"] == OTHER and body["outcome"] == "disagreement"
