"""What the CLI shows of results, batches, predictors and runs: one table row per record that
says which model it is, the results with their intervals, and filters by experiment."""

import json

import pytest
from typer.testing import CliRunner

from histopilot.cli import app
from histopilot.client.transport import Response
from histopilot.commands import output

BATCH_A = "configuration-" + "a" * 64
BATCH_B = "configuration-" + "b" * 64


class Recorder:
    """Answers GETs from a table keyed by path suffix; records every request."""

    def __init__(self, answers):
        self.answers = answers
        self.sent = []

    def send(self, method, path, *, headers, body, timeout):
        if path.endswith("/session"):
            return Response(200, json.dumps({"token": "t"}).encode(), {})
        self.sent.append((method, path))
        for suffix, payload in self.answers.items():
            if path.split("?")[0].endswith(suffix):
                content = json.dumps(payload).encode()
                return Response(200, content, {"content-type": "application/json"})
        return Response(404, json.dumps({"detail": "No.", "code": "NOPE"}).encode(), {})


@pytest.fixture
def answers(monkeypatch):
    from histopilot.commands import common

    table = {}
    monkeypatch.setattr(common, "TRANSPORT_FACTORY", lambda _url: Recorder(table))
    return table


def cli(*arguments):
    result = CliRunner().invoke(app, [*arguments, "--project", "project-1"])
    return result.exit_code, result.stdout


def configuration(model, auroc, interval, ensemble, recalls):
    return {
        "number": 1,
        "model": model,
        "inputMode": "image",
        "seedCount": 3,
        "plannedSeedCount": 3,
        "selected": True,
        "seedAverage": {"auroc": {"mean": auroc, "sd": 0.002, "n": 3}},
        "intervals": {"seedAverage": {"intervals": {"auroc": interval}}},
        "ensemble": {"auroc": ensemble},
        "perClass": [
            {"label": label, "recall": {"mean": value, "sd": 0.01, "n": 3}}
            for label, value in recalls.items()
        ],
    }


EXPERIMENTS = {
    "items": [
        {
            "id": "draft-1",
            "name": "Study",
            "state": "active",
            "batches": [{"id": BATCH_A, "name": "Baseline"}, {"id": BATCH_B, "name": "Other"}],
        }
    ]
}


def test_results_show_intervals_ensembles_class_recall_and_paired_differences(answers):
    answers["/model-experiments/draft-1/results"] = {
        "policy": {"confidenceLevel": 0.95},
        "design": {"resamplingUnit": "slide"},
        "findings": [],
        "batches": [
            {
                "batchId": BATCH_A,
                "name": "Baseline",
                "configurations": [
                    configuration(
                        "abmil",
                        0.812,
                        {"lower": 0.781, "upper": 0.843},
                        0.825,
                        {"high": 0.734, "low": 0.902},
                    )
                ],
            },
            {
                "batchId": BATCH_B,
                "name": "Other",
                "configurations": [
                    configuration(
                        "nnmil", 0.856, {"lower": 0.829, "upper": 0.884}, 0.861, {"high": 0.79}
                    )
                ],
            },
        ],
        "comparisons": [
            {
                "leftBatchId": BATCH_B,
                "rightBatchId": BATCH_A,
                "available": True,
                "oof": {
                    "auroc": {"difference": 0.0441},
                    "balancedAccuracy": {"difference": -0.0127},
                },
                "oofInterval": {
                    "intervals": {
                        "auroc": {"lower": 0.0212, "upper": 0.0673},
                        "balancedAccuracy": {"lower": -0.0418, "upper": 0.0166},
                    }
                },
            }
        ],
    }
    code, text = cli("experiment", "results", "draft-1")
    assert code == 0, text
    header, baseline, other = text.splitlines()[:3]
    assert "95% CI" in header and "SEED ENSEMBLE" in header
    assert "0.812 ± 0.002 (n=3)" in baseline and "0.781–0.843" in baseline
    assert "0.825" in baseline and "0.829–0.884" in other
    assert "  Baseline #1: high 0.734 · low 0.902" in text
    assert "Paired differences, out of fold (95% CI, resampling slides):" in text
    assert "  Other − Baseline: AUROC +0.044 (+0.021 to +0.067)" in text
    assert "balanced accuracy -0.013 (-0.042 to +0.017)" in text


def test_batches_list_with_their_experiment_name_and_execution_state(answers):
    def batch(identity, name):
        spec = {"batchName": name, "experimentId": "draft-1", "experimentName": "Study"}
        manifest = {"kind": "mil-batch", "spec": spec, "experiment": {"id": "draft-1"}}
        return {"id": identity, "manifest": manifest, "createdAt": "2026-01-01T00:00:00Z"}

    answers["/model-experiments"] = EXPERIMENTS
    answers["/mil-experiments/batches"] = {
        "items": [batch(BATCH_A, "Baseline"), batch(BATCH_B, "Other")],
        # A batch that never launched has no execution.
        "executions": [{"batchId": BATCH_A, "status": "completed", "runCounts": {"total": 5}}],
    }
    result = CliRunner().invoke(
        app, ["batch", "list", "--experiment", "study", "--project", "project-1", "--json"]
    )
    rows = json.loads(result.stdout)["data"]
    assert [row["runState"] for row in rows] == ["succeeded", None]
    assert rows[0]["execution"] == {
        "status": "completed",
        "runCounts": {"total": 5},
        "updatedAt": None,
        "finishedAt": None,
    }
    code, text = cli("batch", "list")
    assert code == 0
    first = text.splitlines()[1]
    assert "Baseline" in first and "Study" in first and "succeeded" in first
    # Nor does showing it fail: its list row stands in for the execution it lacks.
    answers[f"/mil-experiments/batches/{BATCH_B}/execution"] = None
    shown = CliRunner().invoke(app, ["batch", "show", BATCH_B, "--project", "project-1", "--json"])
    assert shown.exit_code == 0, shown.stdout
    data = json.loads(shown.stdout)["data"]
    assert data["id"] == BATCH_B and data["runState"] is None


def test_long_names_keep_their_start_and_their_end():
    name = "Inference · Baseline · Study · configuration 1 · seed 42 / split 42 · ensemble"
    shown = output.name(name)
    assert len(shown) == output.MAX_NAME
    assert shown.startswith("Inference · Baseline") and shown.endswith("split 42 · ensemble")
    assert output.cell(shown) == shown
    assert output.name("Short name") == "Short name" and output.name(None) is None
