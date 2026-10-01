"""Agent tools: task-level functions over the client, and the MCP server that offers them."""

import asyncio
import json
import sys

import pytest
from support.cli import Service

from histopilot.agent import tools
from histopilot.client import Client, ClientError
from histopilot.taskcenter.client import default_client


@pytest.fixture
def session(tmp_path, monkeypatch):
    with Service(tmp_path, monkeypatch) as service:
        project = service.create_project()
        created = service.http.post(
            f"/api/v1/projects/{project}/model-experiments",
            json={"name": "Baseline", "operationId": "create:1", "setupVersion": 1},
        )
        assert created.status_code == 201
        # The agent's side, as `agent serve` builds it: a scoped token for the one project.
        exposed = service.http.put(
            f"/api/v1/projects/{project}/ai-exposure",
            json={"level": "full", "expectedLevel": "none"},
        )
        assert exposed.status_code == 200, exposed.text
        token = service.http.post(
            "/api/v1/tokens", json={"projectId": project, "scopes": ["read", "preview"]}
        ).json()["token"]
        client = Client(transport=service.transport, token=token)
        yield tools.AgentSession(client, client.get("/access")), service


def test_reads_page_and_bound_their_answers(session):
    agent, _ = session
    listed = tools.list_records(agent, "experiment")
    assert [item["name"] for item in listed["items"]] == ["Baseline"]
    assert listed["page"] == {"offset": 0, "limit": 50, "hasMore": False}
    experiment = tools.get_record(agent, "experiment", listed["items"][0]["id"])
    assert experiment["runState"] is None
    assert tools.bounded(list(range(250)))[-1] == {
        "truncated": 50,
        "hint": "Ask for a page with offset.",
    }
    with pytest.raises(ValueError):
        tools.list_records(agent, "nonsense")


def test_templates_come_from_the_shared_starters(session):
    agent, _ = session
    design = tools.template(agent, "experiment", "nnmil")
    assert design["batches"][0]["recipe"]["model"] == "nnmil"
    assert tools.template(agent, "experiment")["batches"][0]["id"] == "baseline"
    assert tools.template(agent, "apply")["scope"] == "selected"
    assert tools.template(agent, "targets")["splitUnit"] == "slide"
    with pytest.raises(ValueError, match="experiment templates only"):
        tools.template(agent, "targets", "nnmil")
    with pytest.raises(ValueError, match="unlabeled-cohort"):
        tools.template(agent, "nonsense")


def test_waiting_reports_a_stopped_runner_instead_of_hanging(session):
    agent, service = session
    folder = service.settings.workspace / "projects" / "Study"
    owner = {
        "kind": "experiment",
        "id": "e",
        "projectId": agent.project,
        "projectFolder": str(folder),
        "title": "E",
    }
    task = {
        "id": "task-1",
        "kind": "mil-fold",
        "adapter": "generic",
        "title": "Fold",
        "group": {"kind": "mil-batch", "id": "b"},
        "planOrder": 0,
        "request": {"lane": "cpu"},
        "command": {
            "argv": [sys.executable, "-c", "pass"],
            "cwd": str(folder),
            "log": str(folder / "logs" / "task-1.log"),
        },
    }
    default_client().store.enqueue(owner, [task])
    result = tools.wait(agent, "task", "task-1", timeout_seconds=5)
    # Queued work under a stopped runner has not settled; the wait ends with the reason.
    assert result == {
        "settled": False,
        "outcome": "WORK_NEEDS_ATTENTION",
        "runState": "queued",
        "message": result["message"],
    }
    assert "runner" in result["message"]
    assert agent.client.remaining() is None


# The MCP server -------------------------------------------------------------------------


def names(server):
    return {tool.name for tool in asyncio.run(server.list_tools())}


def test_the_server_offers_only_what_the_token_and_exposure_allow(session):
    pytest.importorskip("mcp")
    from histopilot.agent import server

    agent, _ = session
    offered = names(server.build(agent))
    assert {"status", "list_records", "preview_design", "apply_preview"} <= offered
    agent.access = {**agent.access, "scopes": ["read"]}
    assert not {"preview_design", "apply_preview"} & names(server.build(agent))
    agent.access = {**agent.access, "exposure": "none"}
    assert names(server.build(agent)) == set()
    # No tool can freeze, start, cancel or delete: those need a person.
    annotations = {
        tool.name: tool.annotations for tool in asyncio.run(server.build(session[0]).list_tools())
    }
    assert all(not (hint and hint.destructiveHint) for hint in annotations.values())


def test_tools_run_through_mcp_and_errors_come_back_as_tool_errors(session):
    pytest.importorskip("mcp")
    from histopilot.agent import server

    agent, _ = session
    built = server.build(agent)
    content = asyncio.run(built.call_tool("list_records", {"kind": "experiment"}))
    blocks = content[0] if isinstance(content, tuple) else content
    assert "Baseline" in json.dumps([getattr(block, "text", "") for block in blocks])
    with pytest.raises(Exception) as raised:
        asyncio.run(built.call_tool("get_record", {"kind": "experiment", "identity": "missing"}))
    assert "EXPERIMENT_NOT_FOUND" in str(raised.value)


def test_tool_errors_list_the_findings_behind_them():
    pytest.importorskip("mcp")
    from histopilot.agent import server

    findings = [
        {"code": "F", "message": f"Problem {number}", "severity": "error", "field": "x"}
        for number in range(12)
    ]
    error = ClientError("Blocked.", code="PREVIEW_BLOCKED", kind="refused", findings=findings)
    text = server._error_text(error)
    assert text.splitlines()[0] == "Blocked. [PREVIEW_BLOCKED; refused]"
    assert "- Problem 0 (x) [F]" in text and text.endswith("… and 2 more findings")


def test_the_full_session_token_is_refused(session):
    pytest.importorskip("mcp")
    from histopilot.agent import server

    with pytest.raises(ValueError, match="scoped token"):
        server.connect("http://127.0.0.1:8787", "full-session-token")


def test_a_token_file_of_two_lines_is_refused_without_showing_the_token(tmp_path):
    from typer.testing import CliRunner

    from histopilot.cli import app

    path = tmp_path / "token"
    path.write_text("hpt_0123456789abcdef_secret-part\nsecond line\n", encoding="utf-8")
    result = CliRunner().invoke(app, ["agent", "serve", "--token-file", str(path)])
    assert result.exit_code == 2
    assert "one line" in result.output and "secret-part" not in result.output


def test_status_and_instructions_name_the_one_project_the_token_reaches(session):
    agent, _ = session
    shown = tools.status(agent)
    assert shown["project"] == agent.project and shown["projectName"] == "Study"
    pytest.importorskip("mcp")
    from histopilot.agent import server

    instructions = server.build(agent).instructions
    assert f"one project, Study ({agent.project})" in instructions
    assert "never answer\nabout Study in its place" in instructions


def test_models_and_comparisons_use_the_catalog_names(session):
    agent, _ = session
    names = [item["name"] for item in tools.models(agent)["models"]]
    assert {"abmil", "nnmil", "mean_pool"} <= set(names)
    design = tools.template(agent, "experiment", "baseline", compare_models=["mean_pool"])
    configurations = design["batches"][0]["configurations"]
    assert [arm["model"] for arm in configurations] == ["abmil", "mean_pool"]
    with pytest.raises(ClientError, match="mean_pool"):
        tools.comparison_arms(agent, {"model": "abmil"}, ["mean_pooling"])
    with pytest.raises(ValueError, match="experiment templates only"):
        tools.template(agent, "apply", compare_models=["nnmil"])


def test_client_errors_keep_their_kind(session):
    agent, _ = session
    with pytest.raises(ClientError) as raised:
        tools.get_record(agent, "experiment", "missing")
    assert raised.value.kind == "not-found"


def test_the_server_connects_with_a_scoped_token_and_offers_its_tools(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from histopilot.agent import server

    with Service(tmp_path, monkeypatch) as service:
        project = service.create_project()
        service.http.put(
            f"/api/v1/projects/{project}/ai-exposure",
            json={"level": "metadata", "expectedLevel": "none"},
        )
        token = service.http.post(
            "/api/v1/tokens", json={"projectId": project, "scopes": ["read"], "name": "mcp"}
        ).json()["token"]
        agent = server.connect("http://127.0.0.1:8787", token, transport=service.transport)
        assert agent.project == project and agent.exposure == "metadata"
        offered = names(server.build(agent))
        assert "list_records" in offered and "preview_design" not in offered
        status = asyncio.run(server.build(agent).call_tool("status", {}))
        blocks = status[0] if isinstance(status, tuple) else status
        assert "metadata" in json.dumps([getattr(block, "text", "") for block in blocks])


REQUEST_TOOLS = {"request_freeze", "request_start", "request_apply", "request_task_action"}


def test_request_tools_need_commit_scope_and_say_they_need_approval(session):
    pytest.importorskip("mcp")
    from histopilot.agent import server

    agent, _ = session
    assert not REQUEST_TOOLS & names(server.build(agent))
    assert "cannot freeze" in server.build(agent).instructions
    agent.access = {**agent.access, "scopes": ["read", "preview", "commit"]}
    built = server.build(agent)
    listed = {tool.name: tool for tool in asyncio.run(built.list_tools())}
    assert REQUEST_TOOLS <= set(listed)
    for name in REQUEST_TOOLS:
        assert listed[name].annotations.destructiveHint
        assert listed[name].title == "Needs your approval"
    assert "person must approve" in built.instructions


def test_requests_are_filed_for_a_person_and_change_nothing(tmp_path, monkeypatch):
    with Service(tmp_path, monkeypatch) as service:
        project = service.create_project()
        service.http.put(
            f"/api/v1/projects/{project}/ai-exposure",
            json={"level": "full", "expectedLevel": "none"},
        )
        path = f"/api/v1/projects/{project}/model-experiments"
        created = service.http.post(
            path, json={"name": "Baseline", "operationId": "create:1", "setupVersion": 1}
        ).json()
        token = service.http.post(
            "/api/v1/tokens",
            json={"projectId": project, "scopes": ["read", "preview", "commit"], "name": "mcp"},
        ).json()["token"]
        client = Client(transport=service.transport, token=token)
        agent = tools.AgentSession(client, client.get("/access"))

        filed = tools.request_freeze(agent, created["id"], created["revision"])
        assert filed["state"] == "pending" and filed["requestId"].startswith("request-")
        assert f"histopilot confirm approve {filed['requestId']}" in filed["message"]
        record = service.http.get(f"{path}/{created['id']}").json()
        assert not record.get("frozenSetupId") and record["revision"] == created["revision"]
        pending = service.http.get("/api/v1/agent-requests").json()["requests"]
        assert [(row["id"], row["method"]) for row in pending] == [(filed["requestId"], "POST")]

        with pytest.raises(ClientError) as raised:
            tools.request_task_action(agent, "someone-elses-task", "cancel")
        assert raised.value.code == "TASK_NOT_FOUND"
        with pytest.raises(ValueError):
            tools.request_task_action(agent, "task", "delete")


def test_proposals_follow_the_browser_and_need_the_right_exposure(session):
    agent, _ = session
    experiment = tools.list_records(agent, "experiment")["items"][0]["id"]
    proposed = tools.propose_apply(agent, [experiment])
    assert proposed["spec"]["scope"] == "selected" and proposed["spec"]["predictorIds"] == []
    codes = [item["code"] for item in proposed["findings"]]
    assert codes == ["NO_READY_PREDICTORS", "COHORT_REQUIRED"]
    arms = tools.comparison_arms(agent, {"model": "abmil", "inputMode": "image"}, ["nnmil"])
    assert [arm["model"] for arm in arms["configurations"]] == ["abmil", "nnmil"]
    pytest.importorskip("mcp")
    from histopilot.agent import server

    agent.access = {**agent.access, "exposure": "metadata"}
    offered = names(server.build(agent))
    assert {"propose_apply", "label_sources", "target_from_field"} <= offered
    assert "features_from_extraction" not in offered
    agent.access = {**agent.access, "exposure": "full"}
    assert "features_from_extraction" in names(server.build(agent))


def test_answers_fit_one_budget_and_say_what_was_cut():
    assert tools.fit({"a": 1}) == {"a": 1}
    big = {
        "id": "record-1",
        "rows": [{"n": number, "text": "y" * 200} for number in range(500)],
        "nested": {"blob": {"inner": "z" * 100_000}},
    }
    shaped = tools.fit(big)
    assert len(tools.compact(shaped)) <= tools.RESULT_BUDGET
    assert shaped["id"] == "record-1" and shaped["_note"] == tools.SHORTENED_NOTE
    assert shaped["nested"]["blob"]["inner"].endswith("[98000 more characters]")
    listed = tools.fit({"items": [{"text": "y" * 300} for _ in range(400)]})
    assert listed["_note"].startswith("Lists were cut")
    assert listed["items"][-1]["hint"] == "Ask for a page with offset."


@pytest.mark.parametrize(
    "answer",
    [
        # Many mid-sized strings: none worth shortening, so each is left out in turn.
        {f"k{number}": "x" * 9_000 for number in range(12)},
        # Several long strings: each is shortened once, then left out if still too large.
        {f"k{number}": "x" * 30_000 for number in range(6)},
        # Deep nesting with long keys.
        {"a" * 300: {"b" * 300: {f"c{number}": "y" * 5_000 for number in range(40)}}},
    ],
)
def test_fitting_always_ends_within_the_budget(answer):
    shaped = tools.fit(answer)
    assert len(tools.compact(shaped)) <= tools.RESULT_BUDGET
    assert shaped["_note"]


def test_a_part_of_a_record_comes_back_under_its_own_path(session):
    agent, _ = session
    experiment = tools.list_records(agent, "experiment")["items"][0]["id"]
    part = tools.get_record(agent, "experiment", experiment, path="name")
    assert part == {"name": "Baseline"}


def test_summaries_name_what_they_leave_out():
    record = {
        "id": "run-1",
        "name": "Run",
        "manifest": {"kind": "model-evaluation", "coverage": {"selectedSlideIds": ["s"] * 500}},
        "execution": {"log": "x" * 5000},
    }
    assert tools.summary(record) == {
        "id": "run-1",
        "name": "Run",
        "manifest": {"kind": "model-evaluation"},
        "omitted": ["manifest.coverage", "execution"],
    }
    assert tools.pick(record, "manifest.kind") == "model-evaluation"
    with pytest.raises(ValueError, match="coverage"):
        tools.pick(record, "manifest.nothing")


def test_experiment_results_summarize_seeds_and_intervals():
    seed = {"trainingSeed": 42, "splitSeed": 42, "complete": True}
    results = {
        "experimentId": "draft-1",
        "primaryMetric": "auroc",
        "findings": [],
        "batches": [
            {
                "batchId": "batch-1",
                "name": "Baseline",
                "configurations": [
                    {
                        "candidateId": "candidate-1",
                        "model": "abmil",
                        "seedCount": 2,
                        "seedAverage": {"auroc": {"mean": 0.9, "sd": 0.01, "n": 2}},
                        "intervals": {
                            "unit": "slide",
                            "units": 62,
                            "seedAverage": {"intervals": {"auroc": {"lower": 0.8, "upper": 0.95}}},
                            "ensemble": {"intervals": {"auroc": {"lower": 0.85, "upper": 0.97}}},
                        },
                        "splitSeeds": [
                            {
                                "splitSeed": 42,
                                "seeds": [
                                    {
                                        **seed,
                                        "oof": {"count": 62, "auroc": 0.91, "perClass": [0] * 50},
                                        "folds": [{"fold": fold} for fold in range(5)],
                                    },
                                    {**seed, "trainingSeed": 43, "oof": {"auroc": 0.89}},
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }
    shown = tools._results_summary(results)["batches"][0]["configurations"][0]
    assert shown["seedAverage"]["auroc"]["mean"] == 0.9
    assert shown["intervals"]["seedAverage"] == {"auroc": {"lower": 0.8, "upper": 0.95}}
    assert shown["intervals"]["ensemble"] == {"auroc": {"lower": 0.85, "upper": 0.97}}
    assert [row["trainingSeed"] for row in shown["seeds"]] == [42, 43]
    assert [row["oof"]["auroc"] for row in shown["seeds"]] == [0.91, 0.89]
    # Seed-level detail stays out: each seed's per-class rows and its test folds.
    assert all("perClass" not in row["oof"] for row in shown["seeds"])
    assert '"fold"' not in json.dumps(shown) and "perClass" not in shown
    per_class = [{"label": "high", "support": 20, "recall": {"mean": 0.8, "sd": 0.1, "n": 2}}]
    results["batches"][0]["configurations"][0]["perClass"] = per_class
    kept = tools._results_summary(results)["batches"][0]["configurations"][0]
    assert kept["perClass"] == per_class


def test_case_review_needs_the_full_level(session):
    pytest.importorskip("mcp")
    from histopilot.agent import server

    agent, _ = session
    assert "cases" in names(server.build(agent))
    agent.access = {**agent.access, "exposure": "metadata"}
    assert "cases" not in names(server.build(agent))


def test_answers_come_back_as_compact_json(session):
    pytest.importorskip("mcp")
    from histopilot.agent import server

    agent, _ = session
    content = asyncio.run(server.build(agent).call_tool("list_records", {"kind": "experiment"}))
    blocks = content[0] if isinstance(content, tuple) else content
    assert len(blocks) == 1 and "\n" not in blocks[0].text
    assert json.loads(blocks[0].text)["items"][0]["name"] == "Baseline"


def test_experiments_are_named_by_id_or_by_name(session):
    agent, service = session
    experiment = tools.list_records(agent, "experiment")["items"][0]
    assert tools.get_record(agent, "experiment", "baseline")["id"] == experiment["id"]
    # An unknown name reaches the service as an ID, which answers with its own code.
    with pytest.raises(ClientError) as missing:
        tools.get_record(agent, "experiment", "No such study")
    assert missing.value.code == "EXPERIMENT_NOT_FOUND"
    # Names need not be unique: a shared name is refused with the IDs to choose from.
    created = service.http.post(
        f"/api/v1/projects/{agent.project}/model-experiments",
        json={"name": "BASELINE", "operationId": "create:2", "setupVersion": 1},
    )
    assert created.status_code == 201
    with pytest.raises(ClientError, match="2 experiments are named") as shared:
        tools.get_record(agent, "experiment", "Baseline")
    assert shared.value.code == "EXPERIMENT_AMBIGUOUS"
    assert tools.get_record(agent, "experiment", experiment["id"])["id"] == experiment["id"]


class Canned:
    """A client that answers reads from a table keyed by path suffix and records POSTs."""

    def __init__(self, answers):
        self.answers = answers
        self.posted = []

    def get(self, path, **_options):
        for suffix, payload in self.answers.items():
            if path.split("?")[0].endswith(suffix):
                return payload
        raise ClientError("No.", code="NOT_HERE", kind="not-found")

    def request(self, method, path, *, body=None, **_options):
        self.posted.append((path, body))
        return self.get(path)


def _predictor(identity, batch):
    return {
        "id": identity,
        "lifecycleState": "active",
        "manifest": {
            "name": "Study · config 1 · seed 42 / split 42 · ensemble",
            "experimentId": "draft-1",
            "batchId": batch,
            "candidateId": "c1",
            "candidateNumber": 1,
            "method": "ensemble",
            "trainingSeed": 42,
            "splitSeed": 42,
            "recipe": {"model": batch.removeprefix("b-")},
        },
    }


def _run(identity, predictor, experiment, cohort):
    manifest = {"predictorId": predictor, "experimentId": experiment, "cohortId": cohort}
    return {"id": identity, "manifest": manifest, "execution": {"status": "completed"}}


def test_runs_and_predictors_name_their_model_and_filter_by_experiment_and_cohort():
    # Two models' predictors of one configuration and seeds share a name; `model` tells them
    # apart, as the browser's Runs table does.
    experiments = [
        {
            "id": "draft-1",
            "name": "Study",
            "state": "active",
            "batches": [{"id": "b-abmil", "name": "ABMIL"}, {"id": "b-nnmil", "name": "nnMIL"}],
        },
        {"id": "draft-2", "name": "Other", "state": "active", "batches": []},
    ]
    client = Canned(
        {
            "/model-experiments": {"items": experiments},
            "/predictors": {"items": [_predictor("p-1", "b-abmil"), _predictor("p-2", "b-nnmil")]},
            "/evaluation-runs": {
                "items": [
                    _run("r-1", "p-1", "draft-1", "cohort-1"),
                    _run("r-2", "p-2", "draft-1", "cohort-1"),
                    _run("r-3", "p-1", "draft-2", "cohort-2"),
                ]
            },
            "/evaluation-cohorts": {
                "items": [
                    {"id": "cohort-1", "versionLabel": {"tag": "testing"}},
                    {"id": "cohort-2"},
                ]
            },
            "/evaluation-runs/r-2": _run("r-2", "p-2", "draft-1", "cohort-1"),
        }
    )
    agent = tools.AgentSession(client, {"projectId": "project-1", "scopes": ["read"]})
    listed = tools.list_records(agent, "run", experiment="study", cohort="@testing")["items"]
    assert [row["id"] for row in listed] == ["r-1", "r-2"]
    assert [row["model"]["batch"] for row in listed] == ["ABMIL", "nnMIL"]
    assert [row["model"]["architecture"] for row in listed] == ["abmil", "nnmil"]
    assert listed[1]["model"]["description"] == (
        "nnMIL · Configuration 1 · Fold ensemble · Train 42 / split 42"
    )
    predictors = tools.list_records(agent, "predictor", experiment="draft-1")["items"]
    assert [row["model"]["batch"] for row in predictors] == ["ABMIL", "nnMIL"]
    assert tools.get_record(agent, "run", "r-2")["model"]["predictorId"] == "p-2"
    with pytest.raises(ValueError, match="runs only"):
        tools.list_records(agent, "predictor", cohort="@testing")
    with pytest.raises(ValueError, match="batches, predictors or runs"):
        tools.list_records(agent, "dataset", experiment="Study")
    # A filter that names nothing is refused, so it never reads as "no runs".
    for options, code in (
        ({"experiment": "Unknown study"}, "EXPERIMENT_NOT_FOUND"),
        ({"experiment": "draft-9"}, "EXPERIMENT_NOT_FOUND"),
        ({"cohort": "cohort-9"}, "RECORD_NOT_FOUND"),
    ):
        with pytest.raises(ClientError) as refused:
            tools.list_records(agent, "run", **options)
        assert refused.value.code == code and refused.value.kind == "not-found"


def test_a_summary_compares_two_runs_and_names_their_models():
    summary = {
        "predictorId": "p-1",
        "count": 2,
        "comparison": {"predictorId": "p-2", "agreement": 0.5, "kappa": 0.0, "count": 2},
    }
    client = Canned(
        {
            "/inference/summary": summary,
            "/model-experiments": {"items": []},
            "/predictors": {"items": [_predictor("p-1", "b-abmil"), _predictor("p-2", "b-nnmil")]},
        }
    )
    agent = tools.AgentSession(client, {"projectId": "project-1", "scopes": ["read"]})
    shown = tools.run_analysis(agent, "r-1", "summary", comparison="r-2")
    assert client.posted[0][1] == {"unit": "selected", "comparisonId": "r-2"}
    assert shown["model"]["predictorId"] == "p-1" and shown["model"]["architecture"] == "abmil"
    assert shown["comparison"]["model"]["architecture"] == "nnmil"
    with pytest.raises(ValueError, match="summary analysis"):
        tools.run_analysis(agent, "r-1", "agreement", comparison="r-2")


def test_cases_name_the_run_they_are_compared_with():
    client = Canned({"/cases/query": {"items": [], "total": 0}})
    agent = tools.AgentSession(client, {"projectId": "project-1", "scopes": ["read"]})
    tools.cases(agent, "r-1", outcome="disagreement", comparison="r-2")
    tools.cases(agent, "r-1")
    first, second = (body for _, body in client.posted)
    assert first["comparisonId"] == "r-2" and first["outcome"] == "disagreement"
    assert "comparisonId" not in second
