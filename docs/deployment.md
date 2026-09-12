# Local deployment and development

HistoPilot currently supports a single-user local service bound to loopback. A packaged installation serves the browser UI and `/api/v1` from one Python process. Development uses Vite on port 5173 with its API proxy pointed at the control service on port 8787.

## Development from source

Use Python 3.11+, uv, Node.js 24, and npm. From the repository root:

```bash
uv sync --locked
```

Terminal A:

```bash
uv run histopilot serve --dev --no-browser
```

Terminal B:

```bash
cd web
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`. Vite handles frontend updates and proxies `/api`; `--dev` enables the local development service configuration. Keep API calls through that proxy so the UI obtains its session token from the same origin it uses for requests.

The default Vite proxy expects port 8787. If changing the backend port, update the Vite proxy configuration consistently. The development origin allowlist is limited to the loopback aliases on port 5173; arbitrary origins and wildcard CORS are not supported.

## Build a Python wheel with the UI

From `web/`:

```bash
npm ci
npm run build
```

From the repository root:

```bash
uv run python scripts/bundle_web.py
uv build
```

The bundle script copies the Vite output from `web/dist/` into `histopilot/static/`. The Python wheel includes those compiled assets. Rebuild and bundle after frontend changes before creating a release wheel; a Python build alone does not compile React.

Install the resulting wheel into the intended Python environment, then run:

```bash
histopilot serve
```

Open `http://127.0.0.1:8787`. End users do not need Node.js, npm, or a running Vite server. This describes building/installing the repository artifact; it does not assume a public package release exists.

The root URL opens the start page. Choose **Start a new experiment** to name it and select its exact server storage folder, or **Load an existing experiment** to open a recent entry or saved folder. Data/slide/feature paths and primary settings are optional; settings can be edited later in **MIL experiments**. Creation/loading opens Overview and the existing navigation. `?experiment=<id>#overview` preserves registered experiment context on refresh, and the sidebar experiment button returns to the start page. The CRC KRAS synthetic demo is an explicit separate option.

## Configuration

Defaults:

| Setting | Default |
| --- | --- |
| Config file | `~/.histopilot/config.toml` |
| Bind address | `127.0.0.1` |
| Port | `8787` |
| Workspace | `~/.histopilot/workspace` |
| Allowed data roots | None |

The [example config](../examples/config.toml) uses `[server]` and `[storage]` sections. Command-line workspace, host, port, and data-root settings override the corresponding configuration. Repeated `--data-root` options select the explicit directory allowlist for that launch:

```bash
histopilot serve --workspace /path/to/histopilot-workspace \
  --data-root /mnt/pathology/crc \
  --data-root /mnt/pathology/external-validation \
  --no-browser
```

Use `--config /path/to/config.toml` to read an alternate configuration file. Relative paths in TOML resolve against the configuration file directory; `~` expands to the service user's home directory.

Each data root must refer to a directory on the server. Source browsing (`purpose=source`, the API default) uses only this explicit allowlist; no source root is inferred from the current directory, workspace, or home directory. Storage browsing (`purpose=storage`) includes the application workspace and configured data roots, so a new experiment can be created with default settings before any data is connected.

The exact experiment storage folder must be empty or new, with its parent already present under a permitted storage root. HistoPilot writes `histopilot-project.json` there and indexes it in the central SQLite recent list. Choose a dedicated folder for each experiment. Optional source-directory selection saves read-only references without importing metadata or starting a WSI job; those folders must be under configured data roots.

The CLI acquires an advisory service lock in the workspace and starts one Uvicorn worker. Do not launch several service processes against the same workspace. The initial database schema is initialized on startup with WAL enabled; full migration tooling and background job reconciliation are still planned.

## Remote workstation through SSH

Run the service on the workstation with `--no-browser` and keep it bound to loopback. On your laptop:

```bash
ssh -N -L 8787:127.0.0.1:8787 user@gpu-workstation
```

Then open `http://127.0.0.1:8787` on the laptop. The folder picker still shows configured directories on the workstation. For frontend development, forward port 5173 as well and open the forwarded Vite URL. VS Code's Ports view can provide the same forwarding.

Non-loopback hosts such as `0.0.0.0` are rejected because authenticated network deployment is not implemented. SSH forwarding preserves the supported local service boundary.

For a development preview, an ordinary terminal is sufficient. For a service you intend to keep running through SSH/client disconnects, first check existing sessions:

```bash
tmux ls
```

Create a descriptive session if an existing HistoPilot session is not already running:

```bash
tmux new-session -s histopilot
```

Run the service inside it with the desired workspace/data roots. Detach with `Ctrl+B`, then `D`, and reconnect with `tmux attach -t histopilot`. For important future compute jobs, retain persistent logs and use checkpoint/resume; tmux does not protect against workstation reboot or power loss. These are setup instructions, not a claim that a session is currently running.

## Diagnostics and experiment commands

```bash
histopilot doctor
histopilot doctor --json
histopilot jobs --url http://127.0.0.1:8787
histopilot run examples/crc_kras/experiment-spec.json --validate-only
```

`doctor` reports environment/package metadata without importing PyTorch or initializing CUDA. Package presence does not prove model access, working native libraries, or GPU availability. Isolated NVML and backend execution probes are future work.

`jobs` reads the running service's job endpoint using its local session token. The initial list is empty because worker submission is not implemented. It does not infer jobs from process names or unrelated tmux sessions.

`run` accepts JSON or YAML and validates the same `ExperimentSpec` used by the API. Without `--validate-only`, it reports unavailable execution. A valid schema does not establish accessible WSIs, available features, a trained model, or scientific readiness.

## Validation

From the repository root:

```bash
uv run pytest
```

From `web/`:

```bash
npm run typecheck
npm test
npm run build
```

Browser verification should exercise the packaged UI against the real local API: create an experiment with optional fields blank, confirm its folder descriptor, refresh and reopen it, add source paths/settings, and return to the start page. Separately open the explicit demo to save/reload a cohort and model-run draft and export its specification. Inspect both storage/source-root restrictions and jobs/system state. Real-data/backend tests belong with the first supported compute adapter.

## Local API protections

The service checks allowed loopback Host/port and Origin, rejects cross-site fetches, and uses a random local session token for protected API routes. `/api/v1/session` establishes the token; subsequent API requests carry `X-HistoPilot-Token`. Health checks return only minimal public status. The React client handles the session exchange.

Filesystem browsing resolves paths before enforcing configured roots, including symlink destinations, and bounds directory listings. There is no arbitrary file-content endpoint or WSI upload route. These controls support a local single-user application; institutional authentication, multiuser permissions, HTTPS reverse-proxy deployment, and remote job executors remain future work.
