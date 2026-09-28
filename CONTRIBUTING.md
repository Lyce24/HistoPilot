# Contributing to HistoPilot

This page covers the development environment, the test suites and the rules that keep stored work loadable. For how the application is put together, read [architecture](docs/architecture.md) first.

## Environments

HistoPilot uses [uv](https://docs.astral.sh/uv/) and the committed `uv.lock`. Build environments with `uv sync`; do not `pip install` into them.

| Environment | Command | Used by |
| --- | --- | --- |
| `.venv` (development) | `uv sync --locked --extra training --extra imaging` | Tests, Ruff, and the service during development |
| `.venv-training` | `UV_PROJECT_ENVIRONMENT=.venv-training uv sync --locked --extra training` | Training, refit, evaluation and attention workers started by the service |

The service itself needs only `uv sync --locked`. The `training` extra adds Torch and Lightning, and the `imaging` extra adds Pillow and OpenSlide for slide viewing. Without them, tests that need Torch or Pillow are skipped. The service looks for its worker interpreter in `HISTOPILOT_TRAINING_PYTHON`, then `.venv-training/bin/python` in the checkout; see [deployment](docs/deployment.md#training-environment). A new git worktree has no `.venv-training`, so create one there before running real training.

Frontend dependencies are installed once with `npm --prefix web ci`.

## Python tests

```bash
uv run pytest -n auto --dist worksteal -m "not slow"   # about 3 minutes
uv run pytest -n auto --dist worksteal                 # full suite, about 8 minutes
uv run ruff check .
```

Tests marked `slow` train real models or drive the real Task Center runner. Run the fast tier while you work and the full suite before you hand a change over. CI runs Ruff, the full suite, the frontend tests and build, and a wheel packaging check.

`tests/conftest.py` isolates every test from your machine:

- each test gets its own Task Center state directory (`HISTOPILOT_STATE_DIR`);
- the runner never starts on its own (`HISTOPILOT_TASK_CENTER_AUTOSTART=0`);
- lease and claim registries live in a private temporary directory for the session.

### Testing services that submit work

Services that launch compute queue it in the Task Center, exactly as in production. Use the `task_center` fixture (`tests/support/task_center.py`) to observe and drive that work without a real runner:

```python
def test_submission_queues_one_task_per_fold(task_center, tmp_path):
    ...  # call the service under test
    folds = task_center.tasks(adapter="mil-fold")
    assert len(folds) == 5
    task_center.finish(folds[0]["id"])            # record a successful exit
    task_center.finish(folds[1]["id"], "failed")  # or a failure
```

`Center` offers:

| Helper | Purpose |
| --- | --- |
| `tasks(kind=, adapter=, state=, group=)` | Queued and finished tasks, oldest first |
| `task(id)`, `state(id)` | One task, or just its state |
| `start(id)`, `finish(id, state, returncode=, reason=, error=)` | Move a task through the transitions the runner would record |
| `runner(**options)`, `tick_until(runner, predicate)` | A real runner over this store with a fixed fake host, for tests where the worker itself must run |
| `context(host=)` | The context an adapter hook receives from the runner |

The fixture stops any runner and kills any process group it started when the test ends. Other helpers for common fixtures are in `tests/support/`.

### Stored names must not move

`tests/test_persisted_contracts.py` pins the names that queued tasks and archived code depend on:

- adapter names stored in task rows (`mil-fold`, `compute-job`, `packing` and so on);
- worker modules started with `python -m`, such as `histopilot.workers.managed_fold`;
- worker files started by path, such as the TRIDENT runner;
- the archive protocol marker in `workers/compute_job.py` and `application/experiment_predictors.py`;
- the files that make up the training code fingerprint.

A queued task stores its adapter name and command line, and a submitted experiment runs its follow-up work from an archived copy of the code. Renaming or moving one of these files strands work that already exists on users' machines. If you must move one, add a compatibility shim or a migration, then update the test.

## Frontend

The React UI lives in `web/`:

```bash
cd web
npm test               # Vitest unit and component tests
npx tsc --noEmit       # type check (also: npm run typecheck)
npm run build          # type check and production build
```

`web/scripts/verify-*.mjs` are offline browser checks. Each builds a fixture from the real React components with mocked API responses, drives it in headless Chromium, and starts no HistoPilot server. They need a Playwright Chromium headless shell under `~/.cache/ms-playwright`, or its path in `HISTOPILOT_CHROMIUM`. Run one directly, for example:

```bash
node web/scripts/verify-task-center.mjs
node web/scripts/verify-blca-demo.mjs          # add --readme to refresh docs/assets/blca
```

The service serves the bundle in `histopilot/static/`. `bash serve.sh` rebuilds it when `web/` changed; see [deployment](docs/deployment.md#start-and-restart).

## Documentation and privacy

The GitHub remote is **public**. Docs, tests, fixtures, commit messages and screenshots must not contain study data or results: no cohort names or sizes, real slide or patient identifiers, performance numbers from real studies, reader or consensus details, or paths from a real machine. Use generic placeholders such as `/path/to/slides`, and the synthetic BLCA demo for examples and screenshots.

Keep the docs in `docs/` current rather than adding new review or verification reports. When behaviour changes, update the page that describes it:

| Page | Covers |
| --- | --- |
| [README](README.md) | What HistoPilot is and how to start it |
| [User guide](docs/user-guide.md) | The workflow, stage by stage |
| [Methods](docs/methods.md) | Split, cross-validation, metric and interval definitions |
| [Deployment](docs/deployment.md) | Installation, runtimes, configuration, remote access |
| [Architecture](docs/architecture.md) | Runtime, persistence, execution and the code map |
| [Task Center](docs/task-center.md) | Queue, admission, recovery and the runner |
| [API](docs/api.md) | The local HTTP API |
| [BLCA demo](docs/blca-demo.md) | The synthetic walkthrough |
