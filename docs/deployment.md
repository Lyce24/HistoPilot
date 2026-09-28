# Local deployment and development

HistoPilot currently supports a single-user local service bound to loopback. A packaged installation serves the browser UI and `/api/v1` from one Python process. Development uses Vite on port 5173 with its API proxy pointed at the control service on port 8787.

## Build and run from source

Use Python 3.11+, uv, Node.js 24, and npm. From the repository root:

```bash
uv sync --locked
npm --prefix web ci
```

Then start the server yourself in a normal terminal. The first start builds the frontend:

```bash
bash serve.sh
```

Open `http://127.0.0.1:8787`. Choose **Open BLCA demo** for the bundled synthetic walkthrough, or create/load your own project. The demo requires no source data, model weights, training environment or GPU. Its direct link is `http://127.0.0.1:8787/?project=blca-demo-v1#overview`; see [BLCA demo](BLCA_DEMO.md).

The repository's [`serve.sh`](../serve.sh) checks for `uv` and a prepared Python environment, brings the frontend bundle up to date, then runs the service in the foreground. The bundle in `histopilot/static` records a fingerprint of the `web/` sources it was built from. When `web/` has changed, or no bundle exists, the launcher runs `npm run build` and bundles the result; otherwise it starts at once.

The launcher keeps a few guarantees:
- A failed build never starts the service, and the existing bundle is replaced only after a successful build.
- It never rebuilds while another service from the same checkout is running, because that service serves the same `histopilot/static`. Stop it first, or pass `--no-build` to start with the existing bundle.
- It never installs dependencies. `web/node_modules` may be shared between worktrees.

Missing prerequisites produce setup instructions. `bash serve.sh --help` works before setup and does not start a server. `histopilot serve` started directly prints a warning when its bundle is older than `web/`. `uv run python scripts/bundle_web.py --check` reports the state without changing anything.

Pass normal service options through the launcher:

```bash
bash serve.sh --data-root /path/to/research-data --port 8788
```

From another directory, use `bash /path/to/HistoPilot/serve.sh`. Relative option paths resolve from the repository root. The launcher defaults to `--no-browser`; pass `--browser` to open the URL automatically. It honors `UV_PROJECT_ENVIRONMENT` and uses the existing environment without syncing or downloading. The direct command `uv run histopilot serve --no-browser` remains available.

Server startup and restart are manual. Builds and verification do not launch or restart HistoPilot, and the server is not hosted in tmux. After updating source, stop the existing service in its terminal and start it again with `bash serve.sh` when ready. The launcher rebuilds the frontend if needed, and the service loads the current Python code. Refresh the browser afterward. Avoid starting a second service against the same workspace.

## Frontend development with Vite

For hot reload, use the same installed dependencies and start the backend in Terminal A:

```bash
bash serve.sh --dev
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

The root URL opens the start page. Choose **Start a new project** to name it and select its exact server storage folder, or **Load an existing project** to open a recent entry or saved folder. Optional source paths can be supplied later. Creation/loading opens the project roadmap; each workflow begins with its record library and proceeds through separate review pages. `?project=<id>#overview` preserves project context on refresh, and the active-project button returns to the start page. Older `?experiment=<id>` project links remain supported. **Open BLCA demo** opens a separate read-only example.

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

Start and manage the HistoPilot service in your own terminal. Long-running extraction, packing, training, evaluation and archive work runs through the machine-level [Task Center](TASK_CENTER_DESIGN.md): `histopilot serve` starts its runner in the `hp-runner-<uid>` tmux session (`--no-runner` skips this), and `histopilot runner status|stop` manages it. That is separate from server hosting. Tasks run in their own process groups and survive client disconnection and service or runner restarts; after a workstation reboot, interrupted tasks are queued again (auto-resume, on by default) and resume from their checkpoints when the service starts. Jobs launched before the Task Center keep their own tmux sessions; before manually launching any long-running compute job, check existing sessions to avoid duplicates, retain persistent logs, and use the job's checkpoint/resume support. Updating HistoPilot does not change the code of running workers; `histopilot serve` restarts the runner when its code changed, after the current step.

## Diagnostics

```bash
histopilot doctor
histopilot doctor --json
```

`doctor` reports environment/package metadata without importing PyTorch or initializing CUDA. Package presence does not prove model access, working native libraries, or GPU availability. **System & storage** separately reports available CPU, RAM, GPU, VRAM and disk measurements; module runtime panels probe optional training/extraction dependencies in isolated processes. Missing measurements remain explicitly unavailable.

For saved training runs, resource charts read bounded recorded history. New CPU measurements require a worker version that records CPU counters. Old or already-running worker versions cannot gain historical CPU samples from a UI update; their available GPU/RAM snapshots remain useful. Restart the service manually to load new endpoints, while leaving independent workers under their existing lifecycle controls.

## Validation

From the repository root:

```bash
uv run pytest -n auto --dist worksteal -m "not slow"   # about 2 minutes
uv run pytest -n auto --dist worksteal                 # full suite, about 8 minutes
uv run ruff check .
```

Tests marked `slow` train real models or drive the real Task Center runner. Tests that
need Torch or Pillow skip unless the environment has the `training` and `imaging` extras
(`uv sync --extra training --extra imaging`).

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
