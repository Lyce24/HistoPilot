# Deployment and runtime setup

HistoPilot runs as a single-user service on the machine that holds your data, bound to a loopback address. One Python process serves the browser UI and the `/api/v1` API. Heavy work runs in separate Python environments, started by the [Task Center](task-center.md): feature extraction in a TRIDENT environment, and training, evaluation and attention in a training environment. The service itself never loads Torch, CUDA or model weights.

Storage needs POSIX file locking, directory sync and atomic rename, so run HistoPilot on Linux or WSL; native Windows storage is not supported.

## Install from source

You need Python 3.11+, [uv](https://docs.astral.sh/uv/), Node.js 22.12+ with npm, and tmux for the Task Center runner. From the repository root:

```bash
uv sync --locked
npm --prefix web ci
```

This installs the service into `.venv` and the frontend's build dependencies into `web/node_modules`. Add `--extra imaging` to the `uv sync` if you want to view SVS, TIFF and other OpenSlide slides in the browser. Repeat both commands after an update that changes dependencies. The launcher never installs anything itself.

## Start and restart

```bash
bash serve.sh
```

Open `http://127.0.0.1:8787`. Choose **Open BLCA demo** for the synthetic walkthrough, or start or load a project. Stop the service with **Ctrl+C**.

`serve.sh` does four things in order:

1. It checks that `uv` is on PATH and that the project environment (`$UV_PROJECT_ENVIRONMENT`, else `.venv`) can import HistoPilot. If not, it prints setup instructions.
2. It rebuilds the browser UI when `web/` has changed. The bundle in `histopilot/static/` records a fingerprint of the `web/` sources it was built from; when the fingerprint differs or no bundle exists, the launcher runs `npm run build` and swaps the result in. When nothing changed it starts at once.
3. It starts `histopilot serve` in the foreground, which loads the checkout's current Python code.
4. `histopilot serve` starts the Task Center runner, or restarts it if the runner's code is outdated or it was started from another checkout. The restart waits for the runner's current step; running tasks are adopted, not stopped.

The launcher keeps these guarantees:

- A failed build never starts the service, and the existing bundle is replaced only after a successful build.
- It never rebuilds while another service from the same checkout is running, because that service serves the same bundle. Stop it first, or pass `--no-build` to start with the existing bundle.
- It never installs dependencies. `web/node_modules` may be a link shared between worktrees.

Pass service options through the launcher:

```bash
bash serve.sh --data-root /path/to/research-data --port 8788
```

| Option | Effect |
| --- | --- |
| `--data-root PATH` | Allow a source folder (repeatable). Overrides the configured list. |
| `--workspace PATH` | Application workspace |
| `--config PATH` | Alternate configuration file |
| `--port PORT`, `--host HOST` | Local address. Only `127.0.0.1` and `localhost` are accepted. |
| `--dev` | API auto-reload for frontend development with Vite; skips the UI build |
| `--browser` | Open a browser (the launcher defaults to `--no-browser`) |
| `--no-build` | Serve the existing bundle even if `web/` changed |
| `--no-runner` | Do not start or restart the Task Center runner |

Run the launcher from any directory with `bash /path/to/HistoPilot/serve.sh`; relative option paths resolve from the repository root. `bash serve.sh --help` works before setup. `uv run python scripts/bundle_web.py --check` reports whether the bundle is current without changing it, and `histopilot serve` started directly warns when it is not.

Start and restart the service yourself, in a normal terminal; it is never hosted in tmux. Builds and tests never start or restart it. After updating the source, stop the service and start it again with `bash serve.sh`, then refresh the browser. Running tasks keep the code they started with. Do not start two services against the same workspace; a lock file (`service.lock`) refuses the second.

## Configuration

| Setting | Default |
| --- | --- |
| Configuration file | `~/.histopilot/config.toml` |
| Workspace | `~/.histopilot/workspace` |
| Address | `127.0.0.1:8787` |
| Data roots | None |

The [example configuration](../examples/config.toml) shows the two accepted sections, `[server]` (`host`, `port`, `login`) and `[storage]` (`workspace`, `data_roots`). Unknown keys are rejected. Relative paths resolve from the configuration file's folder, and `~` expands to the service user's home. Command-line options override the file.

```toml
[storage]
data_roots = ["/path/to/research-data", "/path/to/external-validation"]
```

**Data roots** are the only folders the source pickers can browse: metadata tables, slides and features. Nothing is inferred from the current directory or your home. Without data roots you can still create a project and open the demo. The storage picker, used to choose a project folder, also includes the workspace. A new project needs an empty or new folder whose parent exists inside the workspace or a data root. Paths always refer to the machine running the service, even when the browser runs elsewhere.

## Runtime environments

| Environment | Used for | Found through |
| --- | --- | --- |
| Service (`.venv`) | The API, UI, storage and OpenSlide slide viewing | `uv sync --locked`, plus `--extra imaging` for viewing |
| Training | Fold training, refits, evaluations, inference and attention | `HISTOPILOT_TRAINING_PYTHON`, else `<checkout>/.venv-training/bin/python`, else the service's own Python |
| TRIDENT | Feature extraction | `HISTOPILOT_TRIDENT_PYTHON` and `HISTOPILOT_TRIDENT_ROOT`, else the default search below |
| SDPC reader | Reading `.sdpc` slides in the viewer | `HISTOPILOT_SDPC_PYTHON`, else the TRIDENT interpreter |

The Task Center runner inherits `HISTOPILOT_*`, `PATH` and `LD_LIBRARY_PATH` from the service that starts it. Set these variables before `bash serve.sh`, and restart the service (which restarts the runner) after changing them.

### Training environment

Create the training environment once per checkout, from the same lock file:

```bash
UV_PROJECT_ENVIRONMENT=.venv-training uv sync --locked --extra training
```

The `training` extra adds Torch, Lightning, TorchMetrics and scikit-learn. The service finds `.venv-training/bin/python` automatically, and picks up a newly created one without a restart. To use another interpreter, set `HISTOPILOT_TRAINING_PYTHON` before starting the service. A new git worktree has no `.venv-training`, so training from it fails with "No module named 'torch'" until you create one.

The training runtime panel probes this interpreter in a separate process: Torch, Lightning, h5py, PyArrow and scikit-learn imports, versions, CUDA and GPUs. Only the Task Center runner needs tmux; its launcher checks for it. A submitted experiment records the interpreter and package versions it ran with; follow-up work refuses to run in a changed environment (`EXPERIMENT_RUNTIME_CHANGED`). Restore the environment, or copy the experiment to plan it again. See [architecture](architecture.md#pinned-compute-archives).

### TRIDENT feature extraction

Extraction runs TRIDENT in its own Python environment. Point HistoPilot at its interpreter and checkout:

```bash
export HISTOPILOT_TRIDENT_PYTHON=/path/to/trident-environment/bin/python
export HISTOPILOT_TRIDENT_ROOT=/path/to/TRIDENT
bash serve.sh --data-root /path/to/research-data
```

Without these variables HistoPilot searches:

| What | Search order |
| --- | --- |
| Interpreter | `~/miniconda3/envs/trident/bin/python`, `~/anaconda3/envs/trident/bin/python`, then the service's Python |
| Checkout | `<checkout>/.local/TRIDENT` (or `.local/trident`), an editable TRIDENT install in that interpreter, `~/projects/TRIDENT` (or `trident`) |

The first checkout containing `run_batch_of_slides.py` is used. A git worktree has no `.local/TRIDENT` of its own, so set `HISTOPILOT_TRIDENT_ROOT` there. When nothing is found, the extraction preview and the **System & storage** page report TRIDENT as unavailable, list the folders searched and name any sibling checkout that has one. Discovery never runs the interpreter; encoder dependencies, checkpoint access and gated-model logins must work inside that environment and are checked by the worker.

### SDPC slides

SDPC files need OpenSDPC in the TRIDENT interpreter, both for extraction and, by default, for viewing. Install the pinned version into that interpreter; installing it only into HistoPilot's `.venv` does not reach a separate TRIDENT environment:

```bash
uv pip install --python "$HISTOPILOT_TRIDENT_PYTHON" --reinstall-package opensdpc \
  "opensdpc @ git+https://github.com/WonderLandxD/opensdpc@a07579eedde1dffddf8fa712ef236b97ca8cfc55"
```

Reinstalling repairs an editable install whose source folder moved. The same pin is HistoPilot's optional `sdpc` extra (`uv sync --locked --extra sdpc`). On Linux, HistoPilot finds OpenSDPC's bundled native libraries (`LINUX` and `LINUX/ffmpeg`) and adds them to the child process's `LD_LIBRARY_PATH`; no global export is needed.

The viewer reads SDPC slides in an isolated child process using the TRIDENT interpreter. Set `HISTOPILOT_SDPC_PYTHON` to use a separate reader environment; it needs OpenSDPC, Pillow and OpenSlide. Missing dependencies, decoder failures and timeouts appear as recoverable slide-view errors.

For extraction, HistoPilot applies two small runtime fixes to the recognized pinned OpenSDPC reader (no per-tile full garbage collection, and reuse of the open slide handle for metadata). No installed files are changed, and the worker log lists the fixes applied. Set `HISTOPILOT_SDPC_OPTIMIZATIONS=0` to compare against upstream behaviour.

### Environment variables

| Variable | Purpose |
| --- | --- |
| `HISTOPILOT_TRAINING_PYTHON` | Training and compute interpreter |
| `HISTOPILOT_TRIDENT_PYTHON`, `HISTOPILOT_TRIDENT_ROOT` | TRIDENT interpreter and checkout |
| `HISTOPILOT_SDPC_PYTHON` | Interpreter for reading SDPC slides in the viewer |
| `HISTOPILOT_SDPC_OPTIMIZATIONS=0` | Disable the OpenSDPC extraction fixes |
| `HISTOPILOT_STATE_DIR` | Task Center state directory (default `$XDG_STATE_HOME/histopilot`, else `~/.local/state/histopilot`) |
| `HISTOPILOT_TASK_CENTER_AUTOSTART=0` | Never start or restart the runner automatically |
| `HISTOPILOT_CHROMIUM` | Chromium used by the offline browser checks in `web/scripts/` |

## Slide viewing

The viewer reads bounded image regions; it never decodes a whole slide at full resolution. Selecting a slide shows **Preparing slide** while the reader opens and a bounded set of pyramid tiles (at most 128 tiles of 512 pixels, over three zoom levels) is cached, then **Slide ready**. Unvisited high-resolution areas still load on demand.

| Limit | Value |
| --- | --- |
| Reader processes | At most 2 isolated children, shared by OpenSDPC, OpenSlide and the Pillow fallback. Each is recycled after 128 requests, 5 minutes or 60 seconds idle. |
| Native operation deadline | 20 seconds; a hung or crashed child is retired and the next request recovers |
| Child resource limits | 4 GiB address space and 24 MiB output files, on POSIX |
| Server image cache | 128 MiB or 512 lossless PNGs, shared across requests; entries expire after 10 minutes idle |
| Browser cache per viewer | 192 tiles or 256 MiB |
| Image size | 64–2048 pixels per side |

Every image request carries the slide's source fingerprint, so a file that changed on disk is refused rather than mixed into a prepared view.

## Remote workstation through SSH

Run the service on the workstation, then forward the port from your laptop:

```bash
ssh -N -L 8787:127.0.0.1:8787 user@workstation
```

Open `http://127.0.0.1:8787` on the laptop. The folder pickers still browse the workstation. VS Code's Ports view does the same forwarding. Non-loopback hosts such as `0.0.0.0` are refused, because network authentication is not implemented.

Tasks run in their own process groups under the Task Center runner, so closing the SSH session or the browser does not stop them. After a workstation reboot, the runner marks running tasks as interrupted and, with auto-resume on, requeues them from their checkpoints once you start the service again.

## Frontend development with Vite

Terminal A:

```bash
bash serve.sh --dev
```

Terminal B:

```bash
npm --prefix web run dev
```

Open `http://127.0.0.1:5173`. Vite serves the UI with hot reload and proxies `/api` to port 8787; keep API calls on that proxy so the session token comes from the same origin. `--dev` allows the loopback Vite origins on port 5173 only. If you change the backend port, change the proxy in `web/vite.config.ts` too. When forwarding over SSH, forward port 5173 as well.

## Build a wheel

```bash
npm --prefix web ci
npm --prefix web run build
uv run python scripts/bundle_web.py
uv build
```

`scripts/bundle_web.py` copies `web/dist/` into `histopilot/static/`, which the wheel includes along with the synthetic demo resource. A Python build alone does not compile the UI. Install the wheel into an environment and run `histopilot serve --no-browser`; end users then need neither Node.js nor Vite.

## Diagnostics

```bash
histopilot doctor
histopilot doctor --json
histopilot runner status
```

`doctor` reports package metadata without importing Torch or starting CUDA. Package presence does not prove model access, working native libraries or a usable GPU. The **System & storage** page reports CPU, RAM, GPU, VRAM and disk measurements and TRIDENT readiness. The training runtime panel probes the training environment in a separate process.

If the service or runner seems stuck, `kill -USR1 <pid>` writes every thread's stack to `server-<port>-stacks.log` or `runner-stacks.log` in the Task Center state directory.

## Local API protections

- The service accepts only a loopback Host header and refuses cross-site browser requests.
- A random per-process session token from `/api/v1/session` is required on every other API route, in the `X-HistoPilot-Token` header. The UI handles this.
- Folder browsing resolves symlinks before checking the configured roots, and listings are bounded.
- Downloads and slide images come only from validated project artifacts and frozen slide references. There is no general file download or slide upload route.
- Optional sign-in: with `histopilot serve --login`, or `login = true` under `[server]`, the service hands out its session only to a browser that opened the link it printed. That keeps other OS accounts, Windows programs under WSL2 and the far end of an SSH forward from fetching the session. `histopilot login url` prints the link again.
- AI agents get scoped tokens for one project, never the session, and only for projects whose AI-exposure level allows it. Their changes wait for a person's approval, and every request they make is audited. See [AI agents](agents.md).

These controls protect a single-user local service. Shared-lab authentication, per-user permissions, HTTPS reverse proxies and remote job executors are not implemented.
