"""Reads over a run of a predictor on a cohort: scores, agreement, calibration, subgroups,
cases, label-free summaries, exports and paired comparisons. All are read-class routes."""

from .api import Client, segment
from .records import described
from .resources import project_path

UNITS = ("selected", "slide", "patient")
OUTCOMES = (
    "all",
    "error",
    "false_positive",
    "false_negative",
    "correct",
    "unlabeled",
    "disagreement",
)
SORTS = ("confidence_desc", "confidence_asc", "margin_asc", "agreement_asc")


def _run(project: str, run: str) -> str:
    return f"{project_path(project)}/evaluation-runs/{segment(run)}"


def _post(client: Client, path: str, body: dict, **options):
    return client.request(
        "POST", path, body={k: v for k, v in body.items() if v is not None}, **options
    )


def metrics(client: Client, project: str, run: str, *, reference: str | None = None) -> dict:
    return _post(client, f"{_run(project, run)}/scores", {"referenceId": reference})


def agreement(client: Client, project: str, run: str, *, unit: str = "selected") -> dict:
    return _post(client, f"{_run(project, run)}/agreement", {"unit": unit})


def recalibration(client, project, run, *, unit="selected", reference=None) -> dict:
    return _post(
        client, f"{_run(project, run)}/recalibration", {"unit": unit, "referenceId": reference}
    )


def subgroups(client, project, run, *, attribute: str, unit="selected", reference=None) -> dict:
    body = {"unit": unit, "attribute": attribute, "referenceId": reference}
    return _post(client, f"{_run(project, run)}/performance/breakdown", body)


def cases(client: Client, project: str, run: str, query: dict) -> dict:
    return _post(client, f"{_run(project, run)}/cases/query", query)


def cases_csv(client: Client, project: str, run: str, query: dict) -> bytes:
    return _post(client, f"{_run(project, run)}/cases/export", query, accept="bytes")


def inference_summary(client, project, run, *, unit="selected", attribute=None, comparison=None):
    body = {"unit": unit, "attribute": attribute, "comparisonId": comparison}
    return _post(client, f"{_run(project, run)}/inference/summary", body)


def with_models(client: Client, project: str, summary: dict) -> dict:
    """A label-free summary with `model`, its predictor described as `run list` describes
    it, and the same for the run it is compared with."""
    other = summary.get("comparison")
    rows = [{"manifest": {"predictorId": summary.get("predictorId")}}]
    if isinstance(other, dict):
        rows.append({"manifest": {"predictorId": other.get("predictorId")}})
    models = [row["model"] for row in described(client, project, "run", rows)]
    shown = {**summary, "model": models[0]}
    if isinstance(other, dict):
        shown["comparison"] = {**other, "model": models[1]}
    return shown


def predictions_csv(client, project, run, *, unit="selected", attributes=None) -> bytes:
    body = {"unit": unit, "attributes": attributes}
    return client.request(
        "POST", f"{_run(project, run)}/inference/export", body=body, accept="bytes"
    )


def compare(client: Client, project: str, left: str, right: str) -> dict:
    body = {"leftEvaluationId": left, "rightEvaluationId": right}
    return _post(client, f"{project_path(project)}/evaluation-runs/compare", body)
