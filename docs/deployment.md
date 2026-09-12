# Local deployment and development

HistoPilot currently supports a single-user local service bound to loopback. A packaged installation serves the browser UI and `/api/v1` from one Python process. Development uses Vite on port 5173 with its API proxy pointed at the control service on port 8787.

## Build and run from source

Use Python 3.11+, uv, Node.js 24, and npm. From the repository root:

```bash
uv sync --locked
npm --prefix web ci
npm --prefix web run build
uv run python scripts/bundle_web.py
```

Then start the server yourself in a normal terminal:

```bash
uv run histopilot serve --no-browser
```

Open `http://127.0.0.1:8787`. Choose **Open BLCA demo** for the bundled synthetic walkthrough, or create/load your own project. The demo requires no source data, model weights, training environment or GPU. Its direct link is `http://127.0.0.1:8787/?project=blca-demo-v1#overview`; see [BLCA demo](BLCA_DEMO.md).

Server startup and restart are manual. Builds and verification do not launch or restart HistoPilot, and the server is not hosted in tmux. After updating source, rebuild and bundle the frontend, then stop the existing service in its terminal and start it again when ready. Refresh the browser afterward. Avoid starting a second service against the same workspace.

## Frontend development with Vite

For hot reload, use the same installed dependencies and start the backend in Terminal A:

```bash
uv run histopilot serve --dev --no-browser
```

Terminal B:

```bash
npm --prefix web run dev
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
histopilot serve --no-browser
```

Open `http://127.0.0.1:8787`. End users do not need Node.js, npm, or a running Vite server. This describes building/installing the repository artifact; it does not assume a public package release exists.

The root URL opens the start page. Choose **Start a new project** to name it and select its exact server storage folder, or **Load an existing project** to open a recent entry or saved folder. Optional source paths can be supplied later. Creation/loading opens the project roadmap; each workflow begins with its record library and proceeds through separate review pages. `?project=<id>#overview` preserves project context on refresh, and the active-project button returns to the start page. Older `?experiment=<id>` project links remain supported. **Open BLCA demo** opens a separate read-only example; the earlier CRC demo remains compatible through its legacy `synthetic-v1` URL.

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
  --data-root /path/to/research-data \
  --data-root /path/to/external-validation \
  --no-browser
```

Use `--config /path/to/config.toml` to read an alternate configuration file. Relative paths in TOML resolve against the configuration file directory; `~` expands to the service user's home directory.

Each data root must refer to a directory on the server. Replace the example paths with your own directories. Source browsing (`purpose=source`, the API default) uses only this explicit allowlist; no source root is inferred from the current directory, workspace, or home directory. Storage browsing (`purpose=storage`) includes the application workspace and configured data roots, so a new project can be created with default settings before any data is connected. The demo requires no allowed source roots.

The exact project storage folder must be empty or new, with its parent already present under a permitted storage root. HistoPilot writes `histopilot-project.json` there and indexes it in the central SQLite recent list. Choose a dedicated folder for each project. Scientific drafts, frozen versions and model-development experiments belong to that project. Optional source-directory selection saves read-only references without importing metadata or starting a WSI job; those folders must be under configured data roots.

The CLI acquires an advisory service lock in the workspace and starts one Uvicorn worker. Do not launch several service processes against the same workspace. SQLite metadata uses WAL; long-running compute executes separately and retains job receipts, logs and supported checkpoints. See [workspace layout](workspace.md) and [current architecture](ARCHITECTURE.md) for storage and recovery contracts.

## Remote workstation through SSH

Run the service on the workstation with `--no-browser` and keep it bound to loopback. On your laptop:

```bash
ssh -N -L 8787:127.0.0.1:8787 user@gpu-workstation
```

Then open `http://127.0.0.1:8787` on the laptop. The folder picker still shows configured directories on the workstation. For frontend development, forward port 5173 as well and open the forwarded Vite URL. VS Code's Ports view can provide the same forwarding.

Non-loopback hosts such as `0.0.0.0` are rejected because authenticated network deployment is not implemented. SSH forwarding preserves the supported local service boundary.

Start and manage the HistoPilot service in your own terminal. The application uses tmux for supported long-running extraction, packing and training workers; that is separate from server hosting. Before manually launching any long-running compute job, check existing sessions to avoid duplicates, retain persistent logs, and use the job's checkpoint/resume support. tmux protects those workers against client disconnection, not workstation reboot or power loss. Updating HistoPilot does not automatically restart existing workers.

## Diagnostics and experiment commands

```bash
histopilot doctor
histopilot doctor --json
histopilot jobs --url http://127.0.0.1:8787
histopilot run examples/crc_kras/experiment-spec.json --validate-only
```

`doctor` reports environment/package metadata without importing PyTorch or initializing CUDA. Package presence does not prove model access, working native libraries, or GPU availability. **System & storage** separately reports available CPU, RAM, GPU, VRAM and disk measurements; module runtime panels probe optional training/extraction dependencies in isolated processes. Missing measurements remain explicitly unavailable.

`jobs` reads the legacy global job endpoint using its local session token; that endpoint still returns an empty list. For implemented workers, use the browser's compute tray and workflow panels or the specific CLI commands such as `feature-jobs` and `training-status`. The legacy endpoint does not reflect whether training, inference or extraction is running.

`run` accepts JSON or YAML and validates the legacy `ExperimentSpec` used by the demo API. Without `--validate-only`, this command reports unavailable execution. Native ABMIL submission uses a frozen local development batch through the browser or `train-batch`; it is not launched by a demo specification. A valid schema does not establish accessible WSIs, available features, a trained model, or scientific readiness.

For saved training runs, resource charts read bounded recorded history. New CPU measurements require a worker version that records CPU counters. Old or already-running worker versions cannot gain historical CPU samples from a UI update; their available GPU/RAM snapshots remain useful. Restart the service manually to load new endpoints, while leaving independent workers under their existing lifecycle controls.

## Validation

From the repository root:

```bash
uv run pytest
uv run ruff check .
```

From `web/`:

```bash
npm run typecheck
npm test
npm run build
```

Browser verification should exercise record libraries, page transitions, saved-project reopening, dependency review, run details and system-state handling. Use temporary synthetic projects for API checks and the standalone fixtures under `web/scripts/` for browser interactions without a server or research jobs. Verify the BLCA walkthrough separately: all eight modules open from their libraries, navigation stays in the demo, and exported records remain synthetic. A mock browser fixture verifies interface behavior; it does not establish a real training outcome or clinical validity.

Packaging checks should verify that the wheel contains the rebuilt `histopilot/static/` assets and the synthetic resources required by the demo. Source updates, UI builds and tests do not start the server. Keep real manifests, images, features, checkpoints, project folders and local research evidence outside version control.

## Local API protections

The service checks allowed loopback Host/port and Origin, rejects cross-site fetches, and uses a random local session token for protected API routes. `/api/v1/session` establishes the token; subsequent API requests carry `X-HistoPilot-Token`. Health checks return only minimal public status. The React client handles the session exchange.

Filesystem browsing resolves paths before enforcing configured roots, including symlink destinations, and bounds directory listings. Scientific downloads and slide-region routes resolve validated project artifacts; there is no general arbitrary-file download or WSI upload route. These controls support a local single-user application; institutional authentication, multiuser permissions, HTTPS reverse-proxy deployment, and remote job executors remain future work.
