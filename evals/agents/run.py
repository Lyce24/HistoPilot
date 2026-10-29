"""Run real AI agents on the task suite through HistoPilot's MCP server, one run at a time.

Each run gets a fresh copy of the sandbox (fixture.py) served on a private loopback port,
a scoped token, and a Claude Code session in print mode whose only tools are HistoPilot's
MCP tools: built-in tools are off, MCP servers other than HistoPilot's are not loaded, and
no user settings, skills or plugins apply. The service's state is read as a person reads it
before and after the session; checks.py judges the run; one JSON line per run goes to
RESULTS/<campaign>/runs.jsonl with the raw stream next to it. Runs already recorded are
skipped, so a stopped campaign resumes where it stopped.

python evals/agents/run.py build CAMPAIGN
python evals/agents/run.py probe CAMPAIGN --model claude-haiku-5-5
python evals/agents/run.py run CAMPAIGN --model claude-haiku-5-5 --reps 3 [--tasks ID,...]
python evals/agents/run.py rejudge CAMPAIGN   # after fixing a checker bug; say which

Runs use your Claude Code login and count against its plan. Run one campaign at a time.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evals.agents import checks  # noqa: E402

HERE = Path(__file__).resolve().parent
# What a run can change, copied from the template before every run; data/ is never copied.
MUTABLE = ("workspace", "state", "tmp")
TASKS = HERE / "tasks.yaml"
RESULTS = HERE / "results"
PORT = 8797
# The scoped token each task names, as sandbox.json names it, and the token's record name.
TOKENS = {
    "grading": ("grading", "eval: grading"),
    "grading-commit": ("grading", "eval: grading commit"),
    "shared": ("shared", "eval: shared"),
    "public": ("public", "eval: public"),
}
TOOL_PREFIX = "mcp__histopilot__"
# Built-in tools stay off (`--tools ""`); these are refused as well, in case one loads.
# Other MCP servers, such as an account's claude.ai connectors, are kept out by
# `--strict-mcp-config`, and every session's own list of tools is checked before it counts.
DENIED = (
    "Bash",
    "BashOutput",
    "KillShell",
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    "NotebookEdit",
    "Glob",
    "Grep",
    "LS",
    "WebFetch",
    "WebSearch",
    "Task",
    "Agent",
    "ListMcpResourcesTool",
    "ReadMcpResourceTool",
)
# The session's only settings: no memory of earlier sessions, no tools but HistoPilot's.
SETTINGS = {"autoMemoryEnabled": False}
# What the session environment keeps from yours: enough for Claude Code to find its login.
KEPT_ENVIRONMENT = ("HOME", "PATH", "USER", "LOGNAME", "LANG", "LC_ALL", "SHELL", "TZ")
BUDGET_PER_RUN = {"haiku": 0.75, "sonnet": 2.0, "opus": 5.0}
RUN_TIMEOUT = 900
PROBE = (
    "List every tool you can call, by its exact name, one per line, and nothing else. "
    "Do not call any tool."
)


# Campaign files ------------------------------------------------------------------------------


class Campaign:
    def __init__(self, name: str, root: Path = RESULTS, port: int = PORT):
        self.name, self.port = name, port
        self.folder = (root / name).resolve()
        self.sandbox = self.folder / "sandbox"
        self.template = self.folder / "template"
        self.runs = self.folder / "runs.jsonl"
        self.transcripts = self.folder / "transcripts"
        # What each run was judged on, so a fixed checker can judge it again (`rejudge`).
        self.judged = self.folder / "judged"
        self.work = self.folder / "work"
        # Outside any checkout, so the session finds no project files, settings or git.
        self.cwd = Path(tempfile.gettempdir()) / f"histopilot-agent-eval-{name}"

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def study(self) -> dict:
        return json.loads((self.template / "sandbox.json").read_text(encoding="utf-8"))

    def finished(self, retry_errors: bool) -> set[str]:
        done = set()
        if self.runs.exists():
            for line in self.runs.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if record["status"] == "finished" or not retry_errors:
                    done.add(record["runId"])
        return done

    def spent(self) -> float:
        if not self.runs.exists():
            return 0.0
        return sum(
            json.loads(line).get("costUsd") or 0.0
            for line in self.runs.read_text(encoding="utf-8").splitlines()
        )

    def record(self, record: dict) -> None:
        with self.runs.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def build(campaign: Campaign, python: str) -> None:
    """Build the sandbox at its live path once, then keep a copy as the template."""
    if campaign.template.exists():
        print(f"{campaign.template} exists; reusing it.")
        return
    campaign.folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    shutil.rmtree(campaign.sandbox, ignore_errors=True)
    subprocess.run(
        [python, "-W", "ignore", "-m", "evals.agents.fixture", "build", str(campaign.sandbox)]
        + ["--port", str(campaign.port)],
        cwd=ROOT,
        env=_service_environment(campaign),
        check=True,
    )
    campaign.template.mkdir(mode=0o700)
    for child in MUTABLE:
        shutil.copytree(campaign.sandbox / child, campaign.template / child, symlinks=True)
    shutil.copy2(campaign.sandbox / "sandbox.json", campaign.template / "sandbox.json")
    meta = {
        "campaign": campaign.name,
        "builtAt": _now(),
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
    }
    (campaign.folder / "campaign.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def reset(campaign: Campaign) -> None:
    """Fresh copies of the template's workspace, Task Center state and temporary folder at
    the sandbox's own path, so records keep their paths. The data root stays as it was built:
    no tool writes there, and feature sources are pinned by inode and change time, which a
    copy would change."""
    for child in MUTABLE:
        shutil.rmtree(campaign.sandbox / child, ignore_errors=True)
        shutil.copytree(campaign.template / child, campaign.sandbox / child, symlinks=True)


# The service ---------------------------------------------------------------------------------


def _service_environment(campaign: Campaign) -> dict:
    keep = {key: os.environ[key] for key in KEPT_ENVIRONMENT if key in os.environ}
    tmp = campaign.sandbox / "tmp"
    return {
        **keep,
        "HISTOPILOT_STATE_DIR": str(campaign.sandbox / "state"),
        "HISTOPILOT_TASK_CENTER_AUTOSTART": "0",
        "TMPDIR": str(tmp),
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


class Service:
    """The sandbox's service on loopback, for one run; never a runner."""

    def __init__(self, campaign: Campaign, python: str):
        self.campaign, self.python = campaign, python
        self.process = None

    def __enter__(self):
        log = (self.campaign.sandbox / "service.log").open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [self.python, "-W", "ignore", "-m", "evals.agents.fixture", "serve"]
            + [str(self.campaign.sandbox), "--port", str(self.campaign.port)],
            cwd=ROOT,
            env=_service_environment(self.campaign),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("The sandbox service exited; see service.log.")
            try:
                with urllib.request.urlopen(f"{self.campaign.url}/api/v1/health", timeout=2):
                    return self
            except (urllib.error.URLError, OSError):
                time.sleep(0.2)
        raise RuntimeError("The sandbox service did not answer within 60 seconds.")

    def __exit__(self, *_exc):
        if self.process and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)


def snapshot(campaign: Campaign, study: dict) -> dict:
    """The service's state as a person reads it, plus every project's audit log."""
    from histopilot.client import Client, resources
    from histopilot.client import experiments as designs

    client = Client(campaign.url)
    projects = {row["id"]: row["name"] for row in client.get("/projects")["projects"]}
    state = {
        "projectNames": projects,
        "exposure": {
            project: client.get(f"/projects/{project}/ai-exposure")["level"]
            for project in sorted(projects)
        },
        "tokens": [
            {key: row.get(key) for key in ("id", "name", "projectId", "scopes", "revokedAt")}
            for row in client.get("/tokens")["tokens"]
        ],
        "requests": [
            {key: row.get(key) for key in ("id", "projectId", "tokenId", "method", "path", "state")}
            for row in client.get("/agent-requests")["requests"]
        ],
        "experiments": [],
        "tasks": {},
        "taskProjects": {},
    }
    # The sandbox's own projects; the service also lists its read-only demo, which has none.
    for project in sorted(set(study["projects"].values())):
        for row in resources.experiments(client, project, state="all"):
            item = {
                "id": row["id"],
                "projectId": project,
                "name": row.get("name"),
                "revision": row.get("revision"),
                "state": row.get("state"),
                "frozen": bool(row.get("frozenSetupId")),
                "submitted": bool(row.get("submission")),
                **_plans(row.get("batchPlans") or []),
            }
            if row.get("batchPlans") and not item["frozen"] and not item["submitted"]:
                try:
                    item["previewHash"] = designs.preview(client, project, row["id"])["previewHash"]
                except Exception as error:  # noqa: BLE001 - a broken draft has no hash
                    item["previewError"] = str(error)
            state["experiments"].append(item)
    for listing in (None, "history"):
        rows, _ = resources.tasks(client, state=listing, limit=1000)
        for row in rows:
            state["tasks"][row["id"]] = row["state"]
            state["taskProjects"][row["id"]] = (row.get("owner") or {}).get("projectId")
    audit = []
    for log in sorted((campaign.sandbox / "workspace" / "projects").glob("*/audit/actions.jsonl")):
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                audit.append(json.loads(line))
            except ValueError:
                continue
    state["audit"] = audit
    return state


def _plans(plans: list[dict]) -> dict:
    recipes = []
    for plan in plans:
        spec = plan.get("spec") or {}
        configurations = spec.get("configurations") or []
        recipes += [row.get("recipe") or row for row in configurations] or [
            spec.get("recipe") or {}
        ]
    return {
        "models": sorted({recipe.get("model") for recipe in recipes if recipe.get("model")}),
        "recipes": recipes,
    }


# The agent -----------------------------------------------------------------------------------


def agent_command(campaign: Campaign, model: str, prompt: str, mcp_config: Path) -> list[str]:
    budget = next((value for key, value in BUDGET_PER_RUN.items() if key in model), 2.0)
    return [
        shutil.which("claude") or "claude",
        "-p",
        prompt,
        "--model",
        model,
        "--output-format",
        "stream-json",
        "--verbose",
        "--strict-mcp-config",
        "--mcp-config",
        str(mcp_config),
        "--tools",
        "",
        "--allowedTools",
        "mcp__histopilot",
        "--disallowedTools",
        *DENIED,
        "--setting-sources",
        "project",
        "--settings",
        json.dumps(SETTINGS),
        "--permission-mode",
        "default",
        "--permission-prompts",
        "none",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--max-budget-usd",
        str(budget),
    ]


def agent_environment(campaign: Campaign) -> dict:
    """Your login and nothing else: no Claude Code variables of the session that runs this."""
    keep = {key: os.environ[key] for key in KEPT_ENVIRONMENT if key in os.environ}
    return {**keep, "TMPDIR": str(campaign.cwd)}


def run_agent(campaign: Campaign, run_id: str, token: str, model: str, prompt: str) -> dict:
    work = campaign.work / run_id
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(mode=0o700, parents=True)
    state = work / "agent-state"
    state.mkdir(mode=0o700)
    token_file = work / "token"
    descriptor = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(token)
    mcp_config = work / "mcp.json"
    mcp_config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "histopilot": {
                        "type": "stdio",
                        "command": str(ROOT / ".venv-agent" / "bin" / "histopilot"),
                        "args": ["agent", "serve", "--url", campaign.url]
                        + ["--token-file", str(token_file)],
                        "env": {"HISTOPILOT_STATE_DIR": str(state)},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    campaign.cwd.mkdir(mode=0o700, parents=True, exist_ok=True)
    started = time.monotonic()
    process = subprocess.Popen(
        agent_command(campaign, model, prompt, mcp_config),
        cwd=campaign.cwd,
        env=agent_environment(campaign),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=RUN_TIMEOUT)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
    wall = time.monotonic() - started
    campaign.transcripts.mkdir(parents=True, exist_ok=True)
    (campaign.transcripts / f"{run_id}.jsonl").write_text(stdout or "", encoding="utf-8")
    (campaign.transcripts / f"{run_id}.stderr.txt").write_text(stderr or "", encoding="utf-8")
    token_file.unlink(missing_ok=True)
    parsed = parse_stream(stdout or "")
    parsed.update(exitCode=process.returncode, timedOut=timed_out, wallSeconds=round(wall, 2))
    return parsed


def parse_stream(text: str) -> dict:
    """The session's tools, tool calls and results, final answer, cost and usage."""
    init, result = None, None
    calls, trace, answered = {}, [], set()
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        kind = event.get("type")
        if kind == "system" and event.get("subtype") == "init":
            init = {
                "tools": event.get("tools") or [],
                "mcpServers": event.get("mcp_servers") or [],
                "model": event.get("model"),
                "permissionMode": event.get("permissionMode"),
                "slashCommands": event.get("slash_commands") or [],
                "skills": event.get("skills") or [],
                "plugins": event.get("plugins") or [],
                "agents": event.get("agents") or [],
                "claudeCodeVersion": event.get("claude_code_version"),
            }
        elif kind == "assistant":
            for block in (event.get("message") or {}).get("content") or []:
                if block.get("type") == "tool_use":
                    calls[block.get("id")] = {
                        "name": _tool(block.get("name")),
                        "input": block.get("input"),
                    }
        elif kind == "user":
            content = (event.get("message") or {}).get("content")
            for block in content if isinstance(content, list) else []:
                if block.get("type") != "tool_result":
                    continue
                answered.add(block.get("tool_use_id"))
                call = calls.get(block.get("tool_use_id"), {})
                trace.append(
                    {
                        "name": call.get("name"),
                        "input": call.get("input"),
                        "text": _text(block.get("content")),
                        "error": bool(block.get("is_error")),
                    }
                )
        elif kind == "result":
            result = event
    unanswered = [call for key, call in calls.items() if key not in answered]
    result = result or {}
    return {
        "init": init,
        "trace": trace,
        "unanswered": unanswered,
        "answer": result.get("result") or "",
        "resultSubtype": result.get("subtype"),
        "isError": result.get("is_error"),
        "costUsd": result.get("total_cost_usd"),
        "turns": result.get("num_turns"),
        "durationMs": result.get("duration_ms"),
        "durationApiMs": result.get("duration_api_ms"),
        "usage": result.get("usage"),
        "modelUsage": result.get("modelUsage"),
    }


def _tool(name):
    return name[len(TOOL_PREFIX) :] if name and name.startswith(TOOL_PREFIX) else name


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(block.get("text", "") for block in content if isinstance(block, dict))
    return ""


def isolation_problems(init: dict | None) -> list[str]:
    """Anything the session could reach besides HistoPilot's tools."""
    if init is None:
        return ["the session reported no tools (no init event)"]
    problems = [f"tool {name}" for name in init["tools"] if not str(name).startswith(TOOL_PREFIX)]
    problems += [
        f"MCP server {server.get('name')}"
        for server in init["mcpServers"]
        if server.get("name") != "histopilot"
    ]
    problems += [f"skill {item}" for item in init["skills"]]
    # Claude Code's own built-in plugins come with every session; installed ones must not.
    problems += [
        f"plugin {item.get('name')}" for item in init["plugins"] if item.get("path") != "builtin"
    ]
    connected = [server for server in init["mcpServers"] if server.get("name") == "histopilot"]
    if not connected or connected[0].get("status") != "connected":
        problems.append(f"HistoPilot's MCP server is not connected: {connected}")
    return problems


# Campaign loop -------------------------------------------------------------------------------


def load_tasks(selected=None) -> list[dict]:
    import yaml

    tasks = yaml.safe_load(TASKS.read_text(encoding="utf-8"))["tasks"]
    if selected:
        wanted = set(selected)
        unknown = wanted - {task["id"] for task in tasks}
        if unknown:
            raise SystemExit(f"Unknown task IDs: {sorted(unknown)}")
        tasks = [task for task in tasks if task["id"] in wanted]
    return tasks


def one_run(campaign: Campaign, task: dict, model: str, rep: int, python: str) -> dict:
    study = campaign.study()
    project_key, token_name = TOKENS[task["token"]]
    token = study["tokens"][task["token"]]
    token_id = study["answers"]["tokens"][token_name]
    prompt = checks.expand(task["prompt"], study)
    run_id = f"{task['id']}__{model}__r{rep}"
    record = {
        "runId": run_id,
        "campaign": campaign.name,
        "task": task["id"],
        "category": task["category"],
        "token": task["token"],
        "model": model,
        "rep": rep,
        "prompt": prompt,
        "startedAt": _now(),
    }
    reset(campaign)
    try:
        with Service(campaign, python):
            before = snapshot(campaign, study)
            agent = run_agent(campaign, run_id, token, model, prompt)
            after = snapshot(campaign, study)
    except Exception as error:  # noqa: BLE001 - recorded, and the campaign goes on
        return {**record, "status": "error", "error": f"{type(error).__name__}: {error}"}
    problems = isolation_problems(agent["init"]) if agent["init"] else []
    isolation = list(problems)
    run = {
        "answer": agent["answer"],
        "trace": agent["trace"],
        "before": before,
        "after": after,
        "tokens": [token_id],
        "project": study["projects"][project_key],
    }
    verdict = checks.judge(task, run, study)
    campaign.judged.mkdir(parents=True, exist_ok=True)
    (campaign.judged / f"{run_id}.json").write_text(json.dumps(run), encoding="utf-8")
    errored = agent["isError"] or agent["timedOut"] or agent["resultSubtype"] != "success"
    # The model's own failures count: running out of turns or budget. A session that never
    # reached a verdict (no result, an API error, a crash or the harness's timeout) is the
    # harness's failure: recorded as an error, left out of the scores and run again.
    infrastructure = (
        agent["timedOut"]
        or agent["resultSubtype"] in (None, "error_during_execution")
        or (agent["isError"] and agent["resultSubtype"] == "success")
    )
    if infrastructure:
        problems.append(
            f"the session ended without a verdict ({agent['resultSubtype']}, "
            f"timed out: {agent['timedOut']}): {agent['answer'][:200]}"
        )
    return {
        **record,
        "status": "error" if problems or not agent["init"] else "finished",
        "error": "; ".join(problems) or (None if agent["init"] else "no session started"),
        "isolation": isolation,
        "agentError": bool(errored),
        **verdict,
        **{
            key: agent[key]
            for key in (
                "answer",
                "resultSubtype",
                "isError",
                "costUsd",
                "turns",
                "durationMs",
                "durationApiMs",
                "usage",
                "modelUsage",
                "exitCode",
                "timedOut",
                "wallSeconds",
                "unanswered",
            )
        },
        "toolCalls": [
            {"name": item["name"], "input": item["input"], "error": item["error"]}
            for item in agent["trace"]
        ],
        "init": agent["init"],
        "finishedAt": _now(),
    }


def run_campaign(options) -> int:
    campaign = Campaign(options.campaign, port=options.port)
    if not campaign.template.exists():
        raise SystemExit(
            f"Build the sandbox first: python {Path(__file__).name} build {campaign.name}"
        )
    tasks = load_tasks(options.tasks.split(",") if options.tasks else None)
    done = campaign.finished(options.retry_errors)
    planned = [
        (task, model, rep)
        for model in options.model
        for rep in range(1, options.reps + 1)
        for task in tasks
        if f"{task['id']}__{model}__r{rep}" not in done
    ]
    print(f"{len(planned)} runs to go; {len(done)} recorded; spent ${campaign.spent():.2f}.")
    _record_harness(campaign, options)
    for index, (task, model, rep) in enumerate(planned, start=1):
        spent = campaign.spent()
        if options.budget_usd is not None and spent >= options.budget_usd:
            print(f"Stopped: ${spent:.2f} spent of the ${options.budget_usd:.2f} budget.")
            return 3
        record = one_run(campaign, task, model, rep, options.python)
        campaign.record(record)
        print(
            f"[{index}/{len(planned)}] {record['runId']}: {record['status']}"
            + (
                f" success={record.get('success')} violated={record.get('violated')}"
                f" ${record.get('costUsd') or 0:.3f} turns={record.get('turns')}"
                if record["status"] == "finished"
                else f" {record.get('error')}"
            ),
            flush=True,
        )
        if record.get("isolation"):
            # Isolation failed: stop rather than run more sessions with tools they should lack.
            print("Stopped: the session could reach more than HistoPilot's tools.")
            return 4
    return 0


def rejudge(options) -> int:
    """Judge every recorded run again with the current tasks.yaml and checks.py, from what
    the run recorded; print each verdict that changes. Old records are kept beside."""
    campaign = Campaign(options.campaign, port=options.port)
    study = campaign.study()
    tasks = {task["id"]: task for task in load_tasks()}
    records = [json.loads(line) for line in campaign.runs.read_text(encoding="utf-8").splitlines()]
    changed = 0
    for record in records:
        path = campaign.judged / f"{record['runId']}.json"
        if record["status"] != "finished" or not path.exists():
            continue
        verdict = checks.judge(tasks[record["task"]], json.loads(path.read_text()), study)
        if (verdict["success"], verdict["violated"]) != (record["success"], record["violated"]):
            changed += 1
            print(
                f"{record['runId']}: success {record['success']} -> {verdict['success']}, "
                f"violated {record['violated']} -> {verdict['violated']}"
            )
        record.update(verdict)
    backup = campaign.runs.with_name(f"runs.{datetime.now(UTC):%Y%m%dT%H%M%S}.jsonl")
    shutil.copy2(campaign.runs, backup)
    campaign.runs.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in records),
        encoding="utf-8",
    )
    print(f"{changed} verdicts changed; the old records are in {backup.name}.")
    return 0


def _record_harness(campaign: Campaign, options) -> None:
    """What this invocation ran with, appended to harness.jsonl: enough to run it again."""
    version = subprocess.run(
        [shutil.which("claude") or "claude", "--version"], capture_output=True, text=True
    ).stdout.strip()
    entry = {
        "at": _now(),
        "claudeCode": version,
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "models": options.model,
        "reps": options.reps,
        "tasks": options.tasks,
        "budgetUsd": options.budget_usd,
        "command": agent_command(campaign, "MODEL", "PROMPT", Path("mcp.json")),
        "keptEnvironment": [key for key in KEPT_ENVIRONMENT if key in os.environ],
        "timeoutSeconds": RUN_TIMEOUT,
    }
    with (campaign.folder / "harness.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def probe(options) -> int:
    """One session that lists its tools: proof of isolation before a campaign."""
    campaign = Campaign(options.campaign, port=options.port)
    study = campaign.study()
    reset(campaign)
    with Service(campaign, options.python):
        agent = run_agent(campaign, "probe", study["tokens"]["grading"], options.model[0], PROBE)
    problems = isolation_problems(agent["init"])
    print(json.dumps({**(agent["init"] or {}), "answer": agent["answer"]}, indent=2))
    print(f"cost ${agent['costUsd']}, turns {agent['turns']}, isolation problems: {problems}")
    return 1 if problems else 0


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _git(*arguments) -> str:
    try:
        return subprocess.run(
            ["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=("build", "probe", "run", "rejudge"))
    parser.add_argument("campaign")
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--tasks", default=None, help="Comma-separated task IDs (default: all).")
    parser.add_argument("--budget-usd", type=float, default=None, help="Stop once this is spent.")
    parser.add_argument("--retry-errors", action="store_true", help="Run errored runs again.")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument(
        "--python",
        default=str(ROOT / ".venv" / "bin" / "python"),
        help="The service's interpreter (default: .venv).",
    )
    parser.add_argument(
        "--build-python",
        default=str(ROOT / ".venv-training" / "bin" / "python"),
        help="The interpreter that builds the sandbox; the results collection needs torch.",
    )
    options = parser.parse_args(argv)
    options.model = options.model or ["claude-haiku-5-5"]
    if options.action == "build":
        build(Campaign(options.campaign, port=options.port), options.build_python)
        return 0
    if options.action == "probe":
        return probe(options)
    if options.action == "rejudge":
        return rejudge(options)
    return run_campaign(options)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
