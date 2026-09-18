# CI compatibility and manual launcher verification

Local verification passed on September 17, 2026. The reported
[GitHub Actions failure](https://github.com/Lyce24/HistoPilot/actions/runs/35299152589)
failed during `uv run pytest` collection with `ModuleNotFoundError: scripts`.
The [first follow-up run](https://github.com/Lyce24/HistoPilot/actions/runs/35304561074)
passed collection and linting, then exposed four tests that depended on the
runner's actual CPU/RAM capacity (2,300 passed, 36 skipped). Their fake runtimes
now declare their own capacity. See [Check and package](https://github.com/Lyce24/HistoPilot/actions/workflows/checks.yml)
for subsequent remote results.

## Corrections

- **Test imports:** `pythonpath = ["."]` in `pyproject.toml` makes the repository's
  source-only `scripts` namespace importable through the `pytest` console entry
  point, matching `python -m pytest`. Research scripts remain outside the wheel.
- **Demo reproducibility:** explicit `math.fsum` replaces version-dependent
  floating-point summation in synthetic log loss, calibration means and Brier
  score. Strict fixture equality and `--check` pass on Python 3.11 and 3.13.
  Regeneration leaves the packaged JSON byte-for-byte unchanged.
- **Portable process mocks:** recovery tests install their mocked optional
  `signal.pidfd_send_signal` with `raising=False`. Standalone Python builds may
  omit that Linux API; the tests still exercise recovery without signaling real
  workers. No tests were excluded to resolve these failures.
- **Portable resource fixtures:** compute replay, refit, and shared interpretation
  fixtures now supply CPU/RAM capacity alongside their fake executors and runtime.
  This keeps lifecycle and resource-change tests independent of the host running
  pytest. Production resource validation and the requested test resources remain
  unchanged.
- **Manual launcher:** `bash serve.sh` uses the prepared environment and bundled
  frontend, forwards serve options, runs in the foreground, and gives setup
  instructions when inputs are missing. It does not install, build, or restart
  an existing server. Help exits without launching the service.

## Verified results

| Check | Result |
| --- | --- |
| Full Python CI suite, Python **3.11.14**, locked development and imaging dependencies | **2,305 passed, 35 skipped**, zero failures; exit 0; 24 min 3.35 sec |
| Exact `uv run ruff check .` | Passed |
| BLCA bridge, recovery and robustness tests on Python 3.11 | 41 passed |
| Strict BLCA demo suite on Python 3.11.14 and 3.13.9 | 6 passed on each interpreter |
| Clean-source CLI, API, launcher, TRIDENT adapter, BLCA bridge and demo checks | 105 passed, 1 skipped in 16.01 sec |
| Frontend with **Node.js 24.21.0** | `npm ci`, 673 tests across 92 files, TypeScript and Vite production build passed |
| Bundled wheel in an isolated checkout | Frontend `index.html`, both packaged demos and exclusion of research scripts verified |
| Launcher checks | 7 tests using executable stubs, Bash syntax and help passed |
| Four hosted-runner failures with a simulated 2-CPU / 7-GiB host fallback | 4 passed after explicit fake-runtime capacity was supplied |
| Compute and refit modules after the resource-fixture correction | 46 passed on Python 3.11 |
| Shared interpretation/attention fixture consumers with two-core CPU affinity | 100 passed on Python 3.11; 10 additional BLCA analysis checks passed |
| Final source checksums and `git diff --check` | Passed |

The full local Python run preceded the resource-fixture correction; targeted
checks above verify that correction. The full run retained optional-dependency/runtime skips as skips. Its two
warnings are upstream Starlette TestClient deprecations for `httpx` and the
AnyIO `BlockingPortal` alias. The clean-source check omitted local virtual
environments, private project files and the prebuilt frontend; its one skip was
the optional local TRIDENT runtime.

## Reproduction

The full Python run used a separate environment and cache under `/tmp`, preserving
the checkout's existing `.venv` and running environment:

```bash
export UV_PROJECT_ENVIRONMENT=/tmp/histopilot-ci-py311-env
export UV_PYTHON_INSTALL_DIR=/tmp/histopilot-ci-python
export UV_CACHE_DIR=/tmp/histopilot-ci-uv-cache
uv sync --python 3.11 --locked --extra imaging
uv run ruff check .
uv run pytest
uv run python -m histopilot.application.blca_demo --check
```

The final full run used tmux session `hp-ci-py311` and completed successfully.
Its local log is `/tmp/histopilot-ci-py311-pytest.log`; exit status, timestamps and
source checksums use the same prefix. The earlier interrupted run was preserved
separately before restarting with the final changes. Temporary logs are not
included in the repository.

Frontend and packaging checks ran in a disposable source copy with Node.js 24:

```bash
npm --prefix web ci
npm --prefix web test
npm --prefix web run build
uv run python scripts/bundle_web.py
uv build --wheel
bash -n serve.sh
bash serve.sh --help
```

No HistoPilot server was started or restarted during verification. Launcher
tests used stubs, API tests used temporary fixtures, and packaging ran separately
from the existing application environment.
