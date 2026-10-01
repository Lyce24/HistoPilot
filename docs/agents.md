# AI agents

An AI agent can read a HistoPilot project, explain its results and failures, and prepare experiments, cohorts and Apply models runs for you to start. This page explains how to let one in, what it can then see, and why a person still makes every change.

Whatever an agent reads goes to its AI provider, which may keep it. Decide first which projects may be shared at all.

## The one rule: keep private data out of the agent's reach

An agent that can run shell commands runs as you. It can read every file you can, fetch the service's full-access session from `http://127.0.0.1:8787/api/v1/session`, and pass `--yes`. Nothing inside HistoPilot stops it: exposure levels, tokens and approvals below guard against mistakes and against agents without a shell, not against a shell agent on your account.

So run shell agents where private data cannot be reached:

| Setup | Verdict |
| --- | --- |
| A dedicated machine or VM that holds only data you may share | Recommended |
| A container with its own network, workspace, `HISTOPILOT_STATE_DIR`, Task Center runner and data roots, with only shareable data mounted | Acceptable on a shared machine: its loopback is its own, so your private service is out of reach |
| Another OS account on the same machine | Not enough: loopback is shared, so it can fetch your service's session, and files are readable wherever permissions allow |
| Your own account, with a chat app that has only HistoPilot's MCP tools | Holds only while the app has no shell or file tools |

From where the agent will work, check that nothing private is reachable:

```bash
histopilot doctor --agent-isolation --private-url http://127.0.0.1:8787 --private-path /data/private
```

It exits 1 if it can reach a private service or read a private folder.

## What an agent may see: exposure levels

Each project has an AI-exposure level. A new project starts at `none`.

| Level | An agent sees |
| --- | --- |
| `none` | Nothing. No token can be made for the project. |
| `metadata` | Names, designs, statuses and aggregate results. Patient and slide IDs become stable pseudonyms such as `S-FEUXZDNEUS` in ID fields, in lists and maps of cases, and in file names and other text; text keeps all-digit numbers shorter than four characters, which read as counts. Paths, free text people wrote, per-case values (case review included; a list or map of five or more cases shows only its size), raw tables, images, files and exports are withheld. This lowers risk; it is not de-identification. |
| `full` | Everything, including slide images and case-level exports. Use it only for data you are free to send to an AI provider. |

Change it on the **Study backups & sources** page, under **AI agent access**, or with `histopilot project exposure --set metadata`. Raising a project to `full` in the browser asks you to type its name. The top bar's **AI** badge shows the open project's level and links to that panel.

## Tokens

An agent never gets the service's session. It gets a scoped token for one project:

```bash
histopilot token create --name "chat app" --scope read --scope preview --days 7
```

- The token is printed once and stored only as a hash. `histopilot token list` shows every token without its secret, and `histopilot token revoke ID` stops it on its next request.
- Scopes build on each other. `read` reaches reads; `preview` adds previews and saving drafts; `commit` lets the agent ask for changes (see [Approvals](#approvals)). The default is `read` and `preview`.
- A token reaches its own project, the Task Center narrowed to that project's work, and a few machine-wide reads such as `/api/v1/version` and `/api/v1/access`. It can list its project's registered source folders and archive jobs, but never changes them. It never reaches another project, exposure levels, other tokens or the Task Center's capacity and runner, and it cannot fetch the service's own session (`SESSION_NOT_FOR_TOKENS`).
- Files a token's requests read must lie in the project's folder, its registered sources, or a folder its frozen feature versions and bundles use, such as a pack attached from elsewhere.
- Lowering the exposure level or revoking the token takes effect on the agent's next request.

The browser's **AI agent access** panel creates read-and-preview tokens and revokes any token.

## Approvals

With `commit` scope, an agent's change does not run. The service keeps it as a request for 24 hours and answers `202 CONFIRMATION_PENDING` with the request's ID; the CLI exits 7.

```bash
histopilot confirm list
histopilot confirm show request-4f…
histopilot confirm approve request-4f…     # runs it now, as you
histopilot confirm decline request-4f…
```

Approving first claims the request, so it runs once even when two people approve it at the same moment, and a declined or expired request never runs. It then replays the request under your own session, and the service checks it again: a request for an experiment that changed since, or a preview that went stale, is refused. The request records how the replay ended: `approved`, `failed` (the service refused it, so nothing changed) or `unknown` (no answer came, so check the record before asking again). An agent cannot claim or approve its own requests, so a chat app that approves tool calls automatically still changes nothing. The browser lists the same requests under **Requests from AI**.

Without `commit` scope, an agent prepares a change and hands you the command, for example `histopilot experiment freeze EXP --preview-hash HASH`, which runs only if the preview is still the one the agent showed you.

## Connecting an agent

**Claude Code: the plugin.** This repository is also a Claude Code plugin marketplace. `/plugin marketplace add Lyce24/HistoPilot`, then `/plugin install histopilot@histopilot`, installs the MCP server below together with the `histopilot` skill; Claude Code asks for the service URL and the token, and keeps the token in your OS keychain. See [the plugin's README](../plugins/histopilot/README.md).

**Other shell agents.** Set `HISTOPILOT_URL`, `HISTOPILOT_PROJECT` and `HISTOPILOT_TOKEN` (the scoped token), and give the agent the `histopilot` skill in `plugins/histopilot/skills/histopilot/`. The skill tells it to check the exposure level first, to use `--json`, never to pass `--yes`, and to treat text inside records as data.

**Chat apps with MCP tools.** Install the optional `agent` extra into an environment of its own, so the checkout's other environments keep their extras: `UV_PROJECT_ENVIRONMENT=.venv-agent uv sync --locked --extra agent`. Save the token in a file only you can read, and add the server to the app's MCP configuration:

```json
{
  "mcpServers": {
    "histopilot": {
      "command": "/path/to/HistoPilot/.venv-agent/bin/histopilot",
      "args": ["agent", "serve", "--url", "http://127.0.0.1:8787", "--token-file", "/path/to/token"]
    }
  }
}
```

`histopilot agent serve` refuses the full session: it runs only with a scoped token. It offers the tools the token's scopes and the project's level allow, and the service checks every call again:

| Tools | Scope |
| --- | --- |
| `status`, `roadmap`, `list_records`, `get_record`, `template`, `experiment_results`, `run_analysis`, `tasks`, `task`, `wait` | `read` |
| `cases`: each case's label and probabilities | `read`, at the `full` level only |
| `propose_apply`, `apply_configuration`, `label_sources`, `comparison_arms`: the values the browser would propose; `models`: the model names a comparison takes | `read` |
| `features_from_extraction` | `read`, at the `full` level only, since a features spec is a path |
| `preview_design`, `apply_preview`, `target_from_field` | `preview` |
| `request_freeze`, `request_start`, `request_apply`, `request_task_action`, titled "Needs your approval" | `commit` |

Answers are compact JSON that fits what chat apps show inline: `list_records` and `tasks` summarize each item, and `experiment_results` gives each configuration's seed means, intervals, recall and other per-class results averaged over the seeds, the seed ensemble and per-seed results unless asked for `detail="full"`. Experiments can be named by ID or by name. Every predictor and run carries `model`, its experiment, batch, architecture and method, because predictor and run names leave out the batch; `list_records` keeps one experiment's batches, predictors or runs with `experiment`, and the runs on one cohort with `cohort`. `run_analysis` with `summary` and `comparison` gives how often two runs on one cohort agree (kappa, weighted kappa and the cross-table), labels or not; at `full` exposure, `cases` with `comparison` and `outcome="disagreement"` lists the cases where they disagree. An answer still too long is shortened: long texts say how much was cut, and `_omitted` names the parts left out. `get_record` with a `path` reads one part of a record, which comes back under that path. Tools run side by side, so a long `wait` never holds up the others, and a refused call lists the findings behind it.

The server also offers this documentation and the [CLI contract](cli-contract.md) as resources, and prompts for reviewing results, designing an experiment, applying models and triaging tasks.

## Sign-in

`histopilot serve --login` (or `login = true` under `[server]` in the configuration) makes the service hand out its session only to a signed-in browser. It prints a link once and opens it; `histopilot login url` prints it again, and `histopilot login rotate` replaces it, signing every browser out. The CLI signs in by itself as the same OS user, or from `HISTOPILOT_LOGIN`.

Sign-in stops other OS accounts, Windows programs under WSL2 and anyone at the far end of an SSH forward from fetching the session. It does not stop a shell agent on your own account, which can read the link like you.

## The audit log

Each project keeps `audit/actions.jsonl` in its folder: one line per request from a token, allowed or refused, and one per change or administrative request a person makes to the project: under its routes, to its tokens, on its agents' requests (each claim and decision) and on its Task Center work. A line holds the time, the actor, the route, its class and the outcome; a token's line also holds the exposure level (empty when the request was refused before the level was read) and the project and Task Center work the request named. It never holds request bodies or case identifiers. The log rotates at 10 MB and keeps three older files.

The log is evidence against mistakes and against agents without a shell. A shell agent on your account could edit it.

## Testing an agent

`tests/skill_evals/histopilot.md` holds ten graded tasks and a sandbox service seeded with canary identifiers. Among the tasks are a refusal, a prompt injection in an experiment's notes, and an approval that a person declines. `scripts/agent_eval_score.py` scores a transcript for the three safety rules: no change without approval, no answer from another project, and no canary the agent should not have seen.
