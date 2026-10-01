"""The noun-verb commands against an in-process service: envelopes, exit codes, prompts."""

import json
import sys

import pytest
from support.cli import Service

import histopilot
from histopilot.taskcenter.client import default_client


@pytest.fixture
def service(tmp_path, monkeypatch):
    with Service(tmp_path, monkeypatch) as current:
        yield current


def enqueue_task(service, project_id, identity="task-1", log_text="step 1\nstep 2\n"):
    folder = service.settings.workspace / "projects" / "Study"
    (folder / "logs").mkdir(parents=True, exist_ok=True)
    log = folder / "logs" / f"{identity}.log"
    log.write_text(log_text)
    owner = {
        "kind": "experiment",
        "id": "experiment-1",
        "projectId": project_id,
        "projectFolder": str(folder),
        "title": "Experiment one",
    }
    task = {
        "id": identity,
        "kind": "mil-fold",
        "adapter": "generic",
        "title": f"Task {identity}",
        "group": {"kind": "mil-batch", "id": "batch-a"},
        "planOrder": 0,
        "request": {"lane": "gpu", "vramGb": 2.0, "cpuThreads": 2},
        "command": {"argv": [sys.executable, "-c", "pass"], "cwd": str(folder), "log": str(log)},
    }
    default_client().store.enqueue(owner, [task])


def test_status_is_one_envelope_on_stdout(service):
    outcome = service.cli("status", "--json")
    assert outcome.code == 0, outcome.stdout
    envelope = outcome.envelope
    assert envelope["schemaVersion"] == 1 and envelope["ok"] is True
    assert envelope["warnings"] == []
    data = envelope["data"]
    assert data["version"] == histopilot.__version__
    assert data["runner"]["alive"] is False and data["project"] is None


def test_project_lists_are_paged_and_use_saves_a_default(service):
    first, second = service.create_project("Study"), service.create_project("Second")
    listed = service.cli("project", "list", "--json", "--limit", "1").envelope
    assert len(listed["data"]) == 1 and listed["page"] == {"offset": 0, "limit": 1, "hasMore": True}
    ids = {item["id"] for item in service.cli("project", "list", "--json").envelope["data"]}
    assert {first, second} <= ids

    saved = service.cli("use", second, "--json").envelope["data"]
    assert saved == {"url": "http://127.0.0.1:8787", "project": second, "name": "Second"}
    shown = service.cli("project", "show", "--json").envelope["data"]
    assert shown["project"]["id"] == second
    assert service.cli("use", "--json").envelope["data"]["project"] == second
    service.cli("use", "--clear")
    assert service.cli("use", "--json").envelope["data"]["project"] is None


def test_a_command_without_a_project_says_how_to_name_one(service):
    outcome = service.cli("experiment", "list", "--json")
    assert outcome.code == 2
    error = outcome.envelope["error"]
    assert (error["code"], error["kind"], error["status"]) == ("PROJECT_REQUIRED", "invalid", None)


def test_experiments_are_listed_shown_and_summarized(service):
    project = service.create_project()
    created = service.http.post(
        f"/api/v1/projects/{project}/model-experiments",
        json={"name": "Baseline", "operationId": "create:1", "setupVersion": 1},
    )
    assert created.status_code == 201, created.text
    identity = created.json()["id"]

    listed = service.cli("experiment", "list", "--project", project, "--json").envelope["data"]
    assert [(row["id"], row["name"], row["runState"]) for row in listed] == [
        (identity, "Baseline", None)
    ]
    text = service.cli("experiment", "list", "--project", project)
    assert text.code == 0 and "Baseline" in text.stdout and text.stdout.startswith("ID")
    shown = service.cli("experiment", "show", identity, "--project", project, "--json")
    assert shown.envelope["data"]["id"] == identity
    results = service.cli("experiment", "results", identity, "--project", project, "--json")
    assert results.code == 0 and results.envelope["data"]["batches"] == []


def test_missing_records_exit_5_with_the_service_code(service):
    project = service.create_project()
    missing = service.cli("experiment", "show", "nope", "--project", project, "--json")
    assert missing.code == 5
    assert missing.envelope["error"]["code"] == "EXPERIMENT_NOT_FOUND"
    assert missing.envelope["error"]["status"] == 404
    assert service.cli("tasks", "show", "nope", "--json").code == 5


def test_tasks_are_listed_shown_and_their_logs_read(service):
    project = service.create_project()
    enqueue_task(service, project)
    listed = service.cli("tasks", "list", "--project", project, "--json").envelope
    assert [(row["id"], row["state"], row["runState"]) for row in listed["data"]] == [
        ("task-1", "queued", "queued")
    ]
    assert listed["page"]["hasMore"] is False
    everywhere = service.cli("tasks", "list", "--all-projects", "--json").envelope["data"]
    assert [row["id"] for row in everywhere] == ["task-1"]
    shown = service.cli("tasks", "show", "task-1", "--json").envelope["data"]
    assert shown["owner"]["projectId"] == project and shown["runState"] == "queued"
    log = service.cli("tasks", "log", "task-1", "--json").envelope["data"]
    assert log == {"taskId": "task-1", "text": "step 1\nstep 2\n"}
    assert service.cli("tasks", "log", "task-1").stdout == "step 1\nstep 2\n"
    tail = service.cli("tasks", "log", "task-1", "--from", "5", "--json").envelope["data"]
    assert (tail["text"], tail["offset"], tail["nextOffset"], tail["size"]) == (
        "1\nstep 2\n",
        5,
        14,
        14,
    )


def test_following_a_settled_task_streams_json_lines_then_the_envelope(service):
    project = service.create_project()
    enqueue_task(service, project, log_text="done ✓\n")
    assert default_client().store.transition(
        "task-1",
        from_states=("queued",),
        to_state="failed",
        finished_at="2026-09-30T00:00:00+00:00",
        exit={"reason": "error", "returncode": 1},
        detail={"exitReason": "error"},
    )
    lines = service.cli("tasks", "log", "task-1", "--follow", "--json").stdout.splitlines()
    assert json.loads(lines[0]) == {"event": "log", "text": "done ✓\n", "nextOffset": 9}
    envelope = json.loads(lines[-1])
    assert envelope["ok"] is True
    assert envelope["data"] == {
        "taskId": "task-1",
        "state": "failed",
        "runState": "failed",
        "nextOffset": 9,
    }


def test_raw_reads_pass_through_and_raw_commits_need_confirmation(service):
    assert "projects" in service.cli("api", "GET", "/projects", "--json").envelope["data"]
    storage = str(service.settings.workspace / "projects" / "Raw")
    (service.settings.workspace / "projects").mkdir(parents=True, exist_ok=True)
    body = json.dumps({"name": "Raw", "storagePath": storage})

    unconfirmed = service.cli("api", "POST", "/projects", "-d", body, "--json")
    assert unconfirmed.code == 7
    envelope = unconfirmed.envelope
    assert envelope["error"]["code"] == "CONFIRMATION_REQUIRED"
    assert envelope["data"]["preview"]["routeClass"] == "admin"
    assert envelope["data"]["result"] is None

    dry = service.cli("api", "POST", "/projects", "-d", body, "--dry-run", "--json")
    assert dry.code == 0 and dry.envelope["data"]["result"] is None
    names = [item["name"] for item in service.http.get("/api/v1/projects").json()["projects"]]
    assert "Raw" not in names

    confirmed = service.cli("api", "POST", "/projects", "-d", body, "--yes", "--json")
    assert confirmed.code == 0, confirmed.stdout
    assert confirmed.envelope["data"]["result"]["name"] == "Raw"


def test_schemas_list_the_spec_kinds_and_print_their_models(service):
    kinds = [row["kind"] for row in service.cli("schema", "--json").envelope["data"]]
    assert {"targets", "import", "experiment", "cohort"} <= set(kinds)
    targets = service.cli("schema", "targets", "--json").envelope["data"]
    assert "splitUnit" in targets["properties"]
    unknown = service.cli("schema", "nope", "--json")
    assert unknown.code == 2 and unknown.envelope["error"]["code"] == "SPEC_KIND_UNKNOWN"
    # An experiment design's schema says where its inputs and batches are described.
    design = service.cli("schema", "experiment", "--json").envelope["data"]
    assert "experiment-inputs" in design["description"] and "batch" in design["description"]


def test_usage_errors_are_envelopes_in_json_mode(service):
    outcome = service.cli("tasks", "list", "--json", "--bogus")
    assert outcome.code == 2
    error = outcome.envelope["error"]
    assert (error["code"], error["kind"]) == ("USAGE_ERROR", "invalid")
    assert "--bogus" in error["message"]


def test_an_unreachable_or_non_loopback_service_is_named(service, monkeypatch):
    from histopilot.commands import common

    monkeypatch.setattr(common, "TRANSPORT_FACTORY", None)
    unreachable = service.cli("status", "--url", "http://127.0.0.1:9", "--json")
    assert unreachable.code == 6
    assert unreachable.envelope["error"]["code"] == "SERVICE_UNREACHABLE"
    remote = service.cli("status", "--url", "http://example.org:8787", "--json")
    assert remote.code == 2 and remote.envelope["error"]["kind"] == "invalid"


def test_older_commands_keep_their_output(service):
    outcome = service.cli("doctor", "--json")
    assert outcome.code == 0
    assert "python" in json.loads(outcome.stdout) and "schemaVersion" not in outcome.stdout


def test_task_actions_preview_confirm_and_refuse(service):
    project = service.create_project()
    enqueue_task(service, project)

    pending = service.cli("tasks", "cancel", "task-1", "--json")
    assert pending.code == 7
    preview = pending.envelope["data"]["preview"]
    assert preview["action"] == "cancel" and preview["task"]["state"] == "queued"
    assert len(preview["previewHash"]) == 64

    dry = service.cli("tasks", "cancel", "task-1", "--dry-run", "--json")
    assert dry.code == 0 and dry.envelope["data"]["result"] is None
    stale = service.cli("tasks", "cancel", "task-1", "--preview-hash", "0" * 64, "--yes", "--json")
    assert stale.code == 4 and stale.envelope["error"]["code"] == "PREVIEW_CHANGED"
    assert default_client().store.get("task-1")["state"] == "queued"

    confirmed = service.cli(
        "tasks", "cancel", "task-1", "--preview-hash", preview["previewHash"], "--yes", "--json"
    )
    assert confirmed.code == 0, confirmed.stdout
    assert default_client().store.get("task-1")["state"] == "cancelled"

    again = service.cli("tasks", "cancel", "task-1", "--yes", "--json")
    assert again.code == 3
    assert again.envelope["error"]["code"] == "PREVIEW_BLOCKED"
    assert again.envelope["error"]["findings"][0]["code"] == "TASK_ACTION_UNAVAILABLE"
    assert service.cli("tasks", "cancel", "--json").code == 2


def test_owners_are_listed_held_and_released(service):
    project = service.create_project()
    enqueue_task(service, project)
    owners = service.cli("tasks", "owners", "--json").envelope["data"]
    assert [owner["title"] for owner in owners] == ["Experiment one"]
    key = owners[0]["key"]
    held = service.cli("tasks", "hold", key, "--yes", "--json")
    assert held.code == 0, held.stdout
    assert service.cli("tasks", "owner", key, "--json").envelope["data"]["held"] is True
    assert service.cli("tasks", "release", key, "--yes", "--json").code == 0
    assert service.cli("tasks", "owner", key, "--json").envelope["data"]["held"] is False
    assert service.cli("tasks", "move", key, "--to", "sideways", "--json").code == 2


def test_waiting_ends_instead_of_hanging_when_the_runner_is_stopped(service):
    project = service.create_project()
    enqueue_task(service, project)
    outcome = service.cli("tasks", "wait", "task-1", "--json")
    assert outcome.code == 9
    error = outcome.envelope["error"]
    assert (error["code"], error["kind"]) == ("WORK_NEEDS_ATTENTION", "work-failed")
    assert "histopilot runner start" in error["message"]

    store = default_client().store
    store.transition("task-1", from_states=("queued",), to_state="running", started_at="t")
    store.transition("task-1", from_states=("running",), to_state="succeeded", finished_at="t")
    done = service.cli("tasks", "wait", "task-1", "--json")
    assert done.code == 0 and done.envelope["data"]["runState"] == "succeeded"


def test_capacity_is_shown_and_changed_only_with_confirmation(service):
    shown = service.cli("tasks", "capacity", "--json").envelope["data"]
    assert "settings" in shown and "effective" in shown
    pending = service.cli("tasks", "capacity", "--cpu-task-slots", "3", "--json")
    assert pending.code == 7
    assert pending.envelope["data"]["preview"]["change"] == {"cpuTaskSlots": 3}
    changed = service.cli("tasks", "capacity", "--cpu-task-slots", "3", "--yes", "--json")
    assert changed.code == 0, changed.stdout
    assert default_client().store.settings()["cpuTaskSlots"] == 3


def test_a_person_confirms_at_a_terminal_and_no_is_the_default(service, monkeypatch):
    from histopilot.commands import common

    monkeypatch.setattr(common, "interactive", lambda: True)
    project = service.create_project()
    enqueue_task(service, project)
    declined = service.cli("tasks", "cancel", "task-1", input="\n")
    assert declined.code == 7
    assert "Cancel task Task task-1 (queued)." in declined.stderr
    assert "Not confirmed" in declined.stderr
    assert default_client().store.get("task-1")["state"] == "queued"
    # The end of input answers no; Ctrl-C at the prompt interrupts (exit 130).
    assert service.cli("tasks", "cancel", "task-1", input="").code == 7

    def interrupted(*_arguments, **_options):
        try:
            raise KeyboardInterrupt
        except KeyboardInterrupt:
            raise common.typer.Abort() from None

    with monkeypatch.context() as patch:
        patch.setattr(common.typer, "confirm", interrupted)
        assert service.cli("tasks", "cancel", "task-1").code == 130
    assert default_client().store.get("task-1")["state"] == "queued"
    accepted = service.cli("tasks", "cancel", "task-1", input="y\n")
    assert accepted.code == 0, accepted.stderr
    assert default_client().store.get("task-1")["state"] == "cancelled"
    # JSON output never prompts, even at a terminal.
    assert service.cli("tasks", "retry", "task-1", "--json", input="y\n").code == 7


def test_the_runner_entry_point_survives_a_fault_in_the_newer_commands():
    import subprocess

    probe = (
        "import sys, types; "
        "broken = types.ModuleType('histopilot.commands'); "
        "broken.__path__ = []; "
        "sys.modules['histopilot.commands'] = broken; "
        "sys.argv = ['histopilot', 'runner', '--help']; "
        "import runpy; runpy.run_module('histopilot.cli', run_name='__main__')"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "runner" in result.stdout and "noun-verb commands unavailable" in result.stderr


def test_status_names_the_code_on_both_sides(service, monkeypatch):
    from histopilot.client import resources

    envelope = service.cli("status", "--json").envelope
    assert envelope["data"]["serviceRevision"] == envelope["data"]["cliRevision"]
    assert "scoped-tokens" in envelope["data"]["features"] and envelope["warnings"] == []
    # Only the service's side changes: it runs another checkout.
    original = resources.status
    monkeypatch.setattr(
        resources, "status", lambda client: {**original(client), "serviceRevision": "0" * 40}
    )
    skewed = service.cli("status", "--json").envelope
    assert [warning["code"] for warning in skewed["warnings"]] == ["VERSION_SKEW"]


def test_table_cells_cut_long_text_but_never_an_identifier():
    from histopilot.commands import output

    identifier = "configuration-" + "a" * 64
    rows = [{"id": identifier, "name": "A long name " * 8}]
    text = output.table(rows, [("ID", "id"), ("NAME", "name")])
    assert identifier in text and "…" in text


def test_a_fault_while_printing_exits_with_the_contracts_internal_error():
    import typer
    from typer.testing import CliRunner

    from histopilot.commands.common import Result, command

    app = typer.Typer()

    @command(app, "broken")
    def broken(ctx):
        """Prints with a renderer that fails."""
        return Result({"ok": True}, text=lambda: 1 / 0)

    @command(app, "other")
    def other(ctx):
        """A second command, so `broken` keeps its name."""
        return Result({})

    outcome = CliRunner().invoke(app, ["broken"])
    assert outcome.exit_code == 1
    assert "Unexpected CLI failure" in outcome.output and "Traceback" not in outcome.output


def test_experiments_are_named_by_name_and_a_shared_name_is_refused(service):
    project = service.create_project()

    def create(name, operation):
        created = service.http.post(
            f"/api/v1/projects/{project}/model-experiments",
            json={"name": name, "operationId": operation, "setupVersion": 1},
        )
        assert created.status_code == 201, created.text
        return created.json()["id"]

    identity = create("Baseline", "create:1")
    shown = service.cli("experiment", "show", "baseline", "--project", project, "--json")
    assert shown.code == 0 and shown.envelope["data"]["id"] == identity
    results = service.cli("experiment", "results", "Baseline", "--project", project, "--json")
    assert results.code == 0 and results.envelope["data"]["experimentId"] == identity
    second = create("BASELINE", "create:2")
    shared = service.cli("experiment", "show", "Baseline", "--project", project, "--json")
    assert shared.code == 3 and shared.envelope["error"]["code"] == "EXPERIMENT_AMBIGUOUS"
    assert identity in shared.envelope["error"]["message"] and second in shared.stdout
    by_id = service.cli("experiment", "show", second, "--project", project, "--json")
    assert by_id.envelope["data"]["id"] == second
