# Deployment and runtime setup

HistoPilot runs as a single-user service on the machine that holds your data, bound to a loopback address. One Python process serves the browser UI and the `/api/v1` API. Heavy work runs in separate Python environments, started by the [Task Center](task-center.md): feature extraction in a TRIDENT environment, and training, evaluation and attention in a training environment. The service itself never loads Torch, CUDA or model weights.

Storage needs POSIX file locking, directory sync and atomic rename, so run HistoPilot on Linux or WSL; native Windows storage is not supported.

The [container images](#containers) run the same service and Task Center runner without a source checkout.

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
| `--demo` | Serve only the synthetic demo, read-only, to anyone; see [public read-only demo](#public-read-only-demo) |
| `--container` | Inside a container only: listen on all of its interfaces; see [container mode](#container-mode-and-the-local-api-protections) |

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
| `HISTOPILOT_DEMO=1` | Start `histopilot serve` in demo mode, as `--demo` does |
| `HISTOPILOT_LOG_LEVEL`, `HISTOPILOT_LOG_STDERR`, `HISTOPILOT_LOG_FORMAT` | Log level, every record on stderr, and the format of stderr; see [logs](#logs) |
| `HISTOPILOT_CONTAINER=1` | [Container mode](#container-mode-and-the-local-api-protections), as `--container`; the images set it |
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

Open `http://127.0.0.1:8787` on the laptop. The folder pickers still browse the workstation. VS Code's Ports view does the same forwarding. Non-loopback hosts such as `0.0.0.0` are refused, because network authentication is not implemented. The one exception is the [public read-only demo](#public-read-only-demo), which serves nothing from your machine.

Tasks run in their own process groups under the Task Center runner, so closing the SSH session or the browser does not stop them. After a workstation reboot, the runner marks running tasks as interrupted and, with auto-resume on, requeues them from their checkpoints once you start the service again.

## Public read-only demo

`histopilot serve --demo`, or `histopilot serve` with `HISTOPILOT_DEMO=1`, serves the browser UI and the synthetic [BLCA demo](blca-demo.md) to anyone who can reach the port, for a public demo such as a Hugging Face Space. It accepts any `--host`, such as `0.0.0.0`. It reads no configuration file, takes no data roots, sign-in or `--dev`, starts no Task Center runner, and keeps its workspace and Task Center state in a temporary folder that it removes when it stops.

A demo service is usually reached through a reverse proxy, so it trusts nothing about a request: not the client address, Host, Origin, fetch metadata or forwarded headers. It needs to trust nothing, because of what it answers:

- `GET`, `HEAD` and `OPTIONS` only. Any other method is refused with 403 `DEMO_MODE_READ_ONLY` before a route runs.
- Under `/api/`, only the four reads below, which give every caller the same public, synthetic content. Every other API route is refused for every method with 403 `DEMO_MODE_ROUTE_UNAVAILABLE`, including the reads that describe the machine: folders and data roots, the workspace, system and version reports, the Task Center, tokens and agent requests, slide images and exports.
- Outside `/api/`, the packaged UI's files.

| Route | Answers |
| --- | --- |
| `GET /api/v1/health` | Status and version |
| `GET /api/v1/session` | The page's session token. Anyone can fetch it, and it reaches only these reads. |
| `GET /api/v1/projects` | The demo alone, an empty `defaultStoragePath` and `demoOnly: true`, so the start page offers only the demo |
| `GET /api/v1/projects/blca-demo-v1/workspace` | The packaged synthetic fixture |

No other origin is granted a cross-origin read. Pages may be framed by `https://huggingface.co` only, where a Space's page shows the app. A service started without `--demo` keeps every check in [local API protections](#local-api-protections); demo mode changes none of them.

[`deploy/huggingface-space/`](../deploy/huggingface-space/) holds a Docker Space that installs a released wheel and runs the demo; its `PUBLISH.md` lists the steps to create the Space.

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

`scripts/bundle_web.py` copies `web/dist/` into `histopilot/static/`, which the wheel includes along with the synthetic demo resource. A Python build alone does not compile the UI. Install the wheel into an environment and run `histopilot serve --no-browser`; end users then need neither Node.js nor Vite. Each [GitHub release](https://github.com/Lyce24/HistoPilot/releases) attaches the wheel built this way.

## Containers

Two images run HistoPilot from `uv.lock` without a source checkout, on a Linux host with Docker:

| Image | Holds | Runs |
| --- | --- | --- |
| `ghcr.io/lyce24/histopilot` | The service, browser UI and CLI on Python 3.11, without Torch | The synthetic demo, and opening projects and reading results without compute |
| `ghcr.io/lyce24/histopilot-worker` | The same, with the `training` extra (Torch with CUDA 12.8) on an NVIDIA CUDA base | The Task Center runner and every task it starts, and the service of a setup that trains |

Releases are tagged `X.Y.Z`, `X.Y` and `latest`; `main` follows the main branch, and every published image is also tagged `sha-<commit>`, a tag that never moves. Each image is smoke-tested before it is published. `docker/Dockerfile` and `docker/Dockerfile.worker` build them, and `compose.yaml` at the repository root runs them.

### Compose

```bash
echo "HISTOPILOT_DATA=/path/to/research-data" > .env
docker compose up -d
docker compose logs histopilot | grep login=
```

Open the printed `http://127.0.0.1:8787/?login=…` link once; that browser then stays signed in, and `docker compose exec histopilot histopilot login url` prints the link again. This needs an NVIDIA GPU, a driver recent enough for CUDA 12.8, and the NVIDIA Container Toolkit. Without a published image of the chosen version, Compose builds it from the checkout; `docker compose build` does that ahead of time.

| Service | Image | Role |
| --- | --- | --- |
| `histopilot` | worker | The service, published on `127.0.0.1` of the host only, with sign-in. It starts no runner. |
| `worker` | worker | The Task Center runner in the foreground. Every task runs in this container, on its GPUs. |
| `demo` | CPU | `histopilot serve --demo`: the read-only synthetic demo alone. `docker compose up demo` starts only it, with no GPU and nothing mounted. |

Set these variables in an `.env` file beside `compose.yaml`, as above, so that every `docker compose` command mounts the same data and uses the same port.

| Variable | Default | Effect |
| --- | --- | --- |
| `HISTOPILOT_DATA` | `./data` | Folder mounted read-only as `/data`, the only data root |
| `HISTOPILOT_PORT` | `8787` | Port on the host's `127.0.0.1`. The container listens on the same port, because the service accepts only Host headers naming its own port. |
| `HISTOPILOT_VERSION` | `latest` | Image tag |

The service and the runner mount three named volumes at the same paths: `state` is the Task Center state directory (`HISTOPILOT_STATE_DIR`, with the task store and the sign-in secret), `workspace` is the application workspace where new projects go, and `tmp` is `TMPDIR`, where the lease registry and output locks live. Sharing them gives the two containers one queue, one lease registry and one set of locks, as on a workstation. Paths recorded in a project are the container's, such as `/data/slides/…`; a project made in a container opens elsewhere only where those paths exist. **Study backups & sources** exports a project from the workspace volume.

- **Why the service runs from the CUDA image.** Starting training, refits, predictor runs and attention probes the training environment from the service, in a separate process that imports Torch and counts GPUs. The result decides whether runs ask for a GPU, and it is recorded in the experiment's execution contract, which the runner's workers must match. A service without Torch refuses to start training (`TRAINING_RUNTIME_UNAVAILABLE`), and one without GPUs plans CPU runs. Both containers therefore run the same image and see the same GPUs; the service process itself still never imports Torch.
- **Users and files.** Both images run as `histopilot`, user and group 1000, and new volumes take that ownership. Data must be readable by that user; world-readable files are. `--build-arg HISTOPILOT_UID=…` and `HISTOPILOT_GID` build images for another user.
- **Stopping.** Restarting the service leaves tasks running. Stopping the worker stops its tasks with it; with auto-resume on, they are requeued from their checkpoints when it starts again. Restart the runner with `docker compose restart worker`: the Task Center page's restart button works only where the service may start runners itself.
- **Not included.** TRIDENT feature extraction: extract features elsewhere and place them under the data root, or extend the worker image with TRIDENT and set `HISTOPILOT_TRIDENT_PYTHON` and `HISTOPILOT_TRIDENT_ROOT`. SDPC slides: OpenSDPC needs git to install and the system libraries `libgomp1`, `libnuma1` and `libjpeg8` at run time; a derived image can add them and the `sdpc` extra, then set `HISTOPILOT_SDPC_PYTHON`.
- **Command line.** `docker compose exec histopilot histopilot project list` runs the CLI next to the service. A CLI on the host reaches `http://127.0.0.1:8787` too, with `HISTOPILOT_LOGIN` set to the secret in the sign-in link.

### Container mode and the local API protections

Inside a container the service cannot listen on 127.0.0.1 alone: Docker forwards a published port to the container's address on its bridge network, so connections arrive from the bridge rather than from loopback. `histopilot serve --container`, or `HISTOPILOT_CONTAINER=1`, which both images set, makes the service listen on all of the container's interfaces (`0.0.0.0`) instead. It is off by default, it cannot be set in the configuration file, and `--host` still accepts only loopback names. Nothing else changes:

- The service never trusted the peer address. A request must name `127.0.0.1`, `localhost` or `[::1]` with the service's own port as its Host, pass the Origin and fetch-metadata checks, carry the session token and, with `--login`, come from a signed-in browser or a client holding the secret. Container mode leaves each check as it is, so a browser on the host that opens `http://127.0.0.1:8787` is served, while other sites and DNS-rebinding names are refused, as on a workstation.
- `compose.yaml` publishes the port on the host's loopback only (`127.0.0.1:8787:8787`), so the host's network interfaces do not reach it.
- What container mode adds is reach within Docker: other containers on the same Docker network and, depending on the Docker version and the host's firewall, machines on the local network that route to the bridge network can open a connection to the container's port. The Host check stops browsers, not a client that writes its own Host header. The CPU image's default command and `compose.yaml` therefore start the service with `--login`: anything else that reaches the port gets the health check and the static UI, never the session.
- Started without `--login`, container mode prints a warning. So does `--container` outside a container (no `/.dockerenv` or `/run/.containerenv`), where it would open the service to every network interface of the machine.

## Diagnostics

```bash
histopilot doctor
histopilot doctor --json
histopilot runner status
```

`doctor` reports package metadata without importing Torch or starting CUDA. Package presence does not prove model access, working native libraries or a usable GPU. The **System & storage** page reports CPU, RAM, GPU, VRAM and disk measurements and TRIDENT readiness. The training runtime panel probes the training environment in a separate process.

If the service or runner seems stuck, `kill -USR1 <pid>` writes every thread's stack to `server-<port>-stacks.log` or `runner-stacks.log` in the Task Center state directory.

## Logs and metrics

### Logs

The service and the Task Center runner write structured logs, one JSON object per line, to `logs/` in the Task Center state directory:

| File | Holds |
| --- | --- |
| `service-<port>.jsonl` | The service on that port: one access line per request, its own startup and errors, and each task it queues or retries |
| `runner.jsonl` | The runner: each step of a task's life (`queued`, `started`, `stopping`, `finished`, `adopted`) with its kind, adapter, attempt, state, exit code, duration and reason |

Each file rotates at 10 MiB and keeps five older ones (`runner.jsonl.1` and so on). `runner.log` keeps the runner's plain-text messages as before. A worker writes HistoPilot's records to its task's log, the one the Task Center shows; other libraries, such as Lightning, print there as they always have.

A line holds `ts` (UTC), `level`, `logger`, `msg`, `request_id`, `task_id` and `component` (`service`, `runner` or `worker`), then the record's own fields; an error adds its traceback as `exc`. An access line:

```json
{"ts": "2026-10-29T14:03:12.345Z", "level": "INFO", "logger": "histopilot.api.access", "msg": "GET /api/v1/task-center/snapshot 200", "request_id": "4f2c9a1e0b5d4c55a9510f0c4b2f9e1d", "task_id": null, "component": "service", "method": "GET", "route": "/api/v1/task-center/snapshot", "status": 200, "duration_ms": 12.4, "auth": "session"}
```

An access line names the route template, never the values in the path, and the kind of credential presented (`session`, `token` or `none`), never the credential. Queries, headers, cookies and bodies are never logged.

- **Request IDs.** Every response carries `X-Request-ID`: the client's own when it is 1–128 letters, digits and `._:-`, otherwise a new one. Everything logged while the request is served carries it as `request_id`, and so does the line it adds to a project's [audit log](agents.md#the-audit-log). A task that a request queues or retries keeps the ID: the runner hands it to the worker as `HISTOPILOT_REQUEST_ID`, beside `HISTOPILOT_TASK_ID`, and the worker's records carry both.
- **Logging never waits.** Each output has a bounded queue drained by a thread of its own. When a disk stalls or nobody reads the terminal, records are dropped and counted (`histopilot_log_records_dropped_total`); requests are never held up. The service's terminal gets only warnings and errors, such as a port already in use. Uvicorn's startup messages go to the file, and its access log is replaced by the service's own access lines.

| Variable | Effect |
| --- | --- |
| `HISTOPILOT_LOG_LEVEL` | `DEBUG`, `INFO` (default), `WARNING` or `ERROR` |
| `HISTOPILOT_LOG_STDERR=1` | Every record on stderr as well, for containers and log collectors |
| `HISTOPILOT_LOG_FORMAT` | `text` (default, one readable line per record) or `json`: the format of stderr and of workers' records in task logs. The files are always JSON Lines. |

The runner inherits these from the service that starts it, and tasks from the runner. To follow the service's log: `tail -f ~/.local/state/histopilot/logs/service-8787.jsonl | jq -c 'select(.level != "INFO")'`.

### Metrics

`GET /api/v1/metrics` answers in the Prometheus text format, version 0.0.4. Every value is an aggregate: labels name task states, adapters, exit reasons, methods, route templates, status codes and GPU indices, never a project, record, slide, path or token. Task metrics are read from the Task Center store at each scrape, so they survive restarts; request metrics count from the service's start.

| Metric | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `histopilot_tasks` | gauge | `state` | Tasks in each state: `blocked`, `queued`, `starting`, `running`, `stopping`, `succeeded`, `failed`, `cancelled`, `interrupted` |
| `histopilot_tasks_held` | gauge | | Blocked and queued tasks of owners on hold |
| `histopilot_tasks_waiting` | gauge | | Queued tasks the runner left waiting: for capacity, a related task, a busy output or a fenced GPU |
| `histopilot_queue_paused` | gauge | | 1 while the queue is paused |
| `histopilot_tasks_finished_total` | counter | `adapter`, `state` | Task attempts that ended, by how they ended |
| `histopilot_task_failures_total` | counter | `adapter`, `reason` | Attempts that failed, by exit reason: `error`, `oom`, `cuda_failure` and so on |
| `histopilot_task_duration_seconds` | histogram | `adapter` | Run time of each attempt that ended, from 1 second to a day |
| `histopilot_runner_up` | gauge | | 1 while a runner holds the runner lock |
| `histopilot_runner_heartbeat_age_seconds` | gauge | | Seconds since the runner's last heartbeat; it writes one every 2 seconds |
| `histopilot_task_store_up` | gauge | | 0 when the store could not be read; the task metrics are then left out |
| `histopilot_gpu_info` | gauge | `gpu`, `name` | One per GPU that nvidia-smi reports |
| `histopilot_gpu_utilization_ratio`, `histopilot_gpu_memory_used_bytes`, `histopilot_gpu_memory_total_bytes` | gauge | `gpu` | Utilization from 0 to 1, and memory |
| `histopilot_http_requests_total` | counter | `method`, `route`, `status` | Requests this service answered |
| `histopilot_http_request_duration_seconds` | histogram | `method`, `route`, `status` | Time to answer them |
| `histopilot_log_records_dropped_total` | counter | `sink` | Records a log output (`file` or `stderr`) dropped |
| `histopilot_build_info` | gauge | `version`, `revision` | The code the service runs |
| `histopilot_service_start_time_seconds` | gauge | | When the service started |

Failures and finished attempts are counted from the store's event history, and durations from its measurements; neither is pruned, so these counters only grow, and a retry never takes back a failure. GPU readings come from `nvidia-smi` (on WSL, `/usr/lib/wsl/lib/nvidia-smi`), asked at most every 5 seconds with a short timeout; without it, or when it fails, the GPU metrics are left out.

**Who may read them.** The metrics are machine-wide, like the Task Center's capacity. The service's session token reads them. A scraper needs no session: reaching the loopback port is enough, as it is to fetch the session itself. A service started with `--login` asks a scraper for its sign-in secret, as a bearer token or in `X-HistoPilot-Login`. Agent tokens are refused with 403 `SCOPE_MISSING`, and the attempt is audited.

```yaml
# prometheus.yml
scrape_configs:
  - job_name: histopilot
    metrics_path: /api/v1/metrics
    scrape_interval: 30s
    static_configs:
      - targets: ["127.0.0.1:8787"]
    # Only for a service started with --login: the secret's file in the Task Center state
    # directory. Only your user can read it, so run Prometheus as you or give it a copy.
    authorization:
      credentials_file: /path/to/histopilot-state/login-8787.secret
```

## Local API protections

- The service accepts only a loopback Host header and refuses cross-site browser requests.
- A random per-process session token from `/api/v1/session` is required on every other API route, in the `X-HistoPilot-Token` header. The UI handles this. A metrics scraper is the one exception; see [metrics](#metrics).
- Folder browsing resolves symlinks before checking the configured roots, and listings are bounded.
- Downloads and slide images come only from validated project artifacts and frozen slide references. There is no general file download or slide upload route.
- Optional sign-in: with `histopilot serve --login`, or `login = true` under `[server]`, the service hands out its session only to a browser that opened the link it printed. That keeps other OS accounts, Windows programs under WSL2 and the far end of an SSH forward from fetching the session. `histopilot login url` prints the link again.
- AI agents get scoped tokens for one project, never the session, and only for projects whose AI-exposure level allows it. Their changes wait for a person's approval, and every request they make is audited. See [AI agents](agents.md).
- In a container the service listens on the container's interfaces, and every check above still applies; see [container mode](#container-mode-and-the-local-api-protections).

These controls protect a single-user local service. Shared-lab authentication, per-user permissions, HTTPS reverse proxies and remote job executors are not implemented. A [demo service](#public-read-only-demo) replaces them with its own, narrower boundary.
