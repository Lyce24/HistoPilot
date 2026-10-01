# HistoPilot for Claude Code

This plugin lets Claude Code work with a HistoPilot service you run yourself. It adds:

- **An MCP server** (`histopilot agent serve`) whose tools read projects, cross-validated results, predictor runs and the Task Center, and prepare experiments and Apply models runs.
- **The `histopilot` skill**, which tells Claude how to use those tools and the `histopilot` CLI safely.

Claude never changes a study by itself. With a commit-scope token its change becomes a request that you approve (`histopilot confirm approve ID`, or **Requests from AI** in the browser); without one, it hands you the command to run.

## Before you install

1. **A running service.** Start HistoPilot with `bash serve.sh` on this machine, or forward its port over SSH.
2. **A project shared with AI agents.** Every project starts at exposure level `none`. On **Study backups & sources → AI agent access**, or with `histopilot project exposure --set metadata`, choose:
   - `metadata`: names, designs, statuses and aggregate results; patient and slide IDs become pseudonyms; files, images, case-level values and exports stay with the service.
   - `full`: everything, including case-level results. Use it only for data you are free to send to an AI provider.
3. **A token for that project**, printed once:

   ```bash
   histopilot token create --name "Claude Code" --scope read --scope preview --days 7
   ```

   Add `--scope commit` to let Claude file requests for you to approve.
4. **A way to run the server**, one of:
   - [uv](https://docs.astral.sh/uv/) on your PATH: the plugin runs HistoPilot from GitHub with `uvx`. The first start downloads it and can take a minute.
   - A HistoPilot checkout with the agent extra: `UV_PROJECT_ENVIRONMENT=.venv-agent uv sync --locked --extra agent`, then give the plugin its `.venv-agent/bin/histopilot`.

## Install

In Claude Code:

```text
/plugin marketplace add Lyce24/HistoPilot
/plugin install histopilot@histopilot
```

When you enable it, Claude Code asks for:

| Setting | What to enter |
| --- | --- |
| HistoPilot service URL | Your service, such as `http://127.0.0.1:8788`. Empty: `HISTOPILOT_URL`, else `http://127.0.0.1:8787` |
| Agent token | The `hpt_…` token. Claude Code keeps it in your OS keychain |
| histopilot executable | Optional: a checkout's `.venv-agent/bin/histopilot`. Empty: `uvx` |

Change them later with `/plugin configure histopilot`. Then ask, for example, "Summarize the experiments in my HistoPilot project and their results."

To try the plugin from a checkout without installing it, set `HISTOPILOT_URL`, `HISTOPILOT_TOKEN` and `HISTOPILOT_AGENT_COMMAND` (the path of a `histopilot` with the agent extra) in your shell, then start Claude Code with `claude --plugin-dir plugins/histopilot`.

## Privacy

Everything a tool returns enters Claude's context and goes to Anthropic. The exposure level decides what that is, and the service enforces it for every request: a token reaches one project, cannot fetch the service's own session, and cannot change exposure levels or tokens.

A Claude Code session also has a shell, and a shell on your account can read your files and reach your service without the token. Run Claude Code with this plugin on a machine or container that holds only data you may share, as [AI agents](../../docs/agents.md) explains.
