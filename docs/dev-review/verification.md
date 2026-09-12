# Development review verification

Verified 2026-09-12 in `/home/yc_liu/projects/HistoPilot-dev`, branch `HistoPilot-dev`.
Review entry point: [verdict](../DEV_REVIEW.md).

## Results

| Check | Result | Scope |
| --- | --- | --- |
| Full backend integration suite | **1,448 passed, 16 skipped**, 644.90 seconds | Complete suite collected during integration. Two dependency deprecation warnings. |
| Final affected backend suites | **200 passed, 9 skipped**, 194.20 seconds | Rerun after the final launch-inspection race and metadata-depth fixes; includes API/System, execution, extraction, packing and locks. Overlaps the full suite. |
| Optional ML runtime | **125 passed**, 158.40 seconds | Synthetic CPU MIL data/model/training, refit, inference and attention. Uses the installed training interpreter with the dev checkout on PYTHONPATH. |
| Frontend | **372 passed in 54 files** | Vitest unit, API contract and rendered-markup regressions. |
| TypeScript and production UI build | **Passed** | Entry remains approximately 909 kB minified / 242 kB gzip; Vite reports its existing large-chunk warning. |
| Offline browser interactions | **21 passed** | Real development React components, mocked APIs and invented data; desktop/mobile layout, controls, retry identities and render recovery. |
| Ruff and whitespace | **Passed** | Repository lint and `git diff --check`; changed Python files checked for formatting. |
| Python wheel | **Built and inspected** | Includes the four compiled frontend assets and demo resource; packaged Python and frontend files compared byte-for-byte with the dev checkout. |

Counts in different rows overlap; do not add them as unique tests. Skipped control-environment tests were not counted as passing. The separate ML runtime covers the selected optional training modules, not every optional dependency or hardware path. ML warnings concern test checkpoint reuse, small synthetic metric samples, loader configuration and dependency behavior; they are retained in the log.

The first in-sandbox backend run hung in Starlette TestClient/AnyIO startup and was stopped. A single identical test passed outside the sandbox in 0.96 seconds. Both backend integration commands above then ran outside that sandbox with temporary test projects. No running HistoPilot service was required or started.

## Commands and logs

The shell initially selected Node 18, which cannot start this repository's Vite/Vitest. UI checks used the existing supported **Node 22.23.2** by prepending `/home/yc_liu/.nvm/versions/node/v22.23.2/bin` to PATH. The control interpreter is Python 3.13.9. No dependency versions or lockfiles were changed.

```bash
# Full backend (normal terminal; bounded test run, no tmux job)
timeout 840s .venv/bin/python -m pytest -q --durations=15 -o faulthandler_timeout=90

# Rerun all affected execution/API suites after final backend corrections
.venv/bin/python -m pytest -q \
  tests/test_compute_jobs.py tests/test_worker_robustness.py \
  tests/test_training_execution.py tests/test_project_lock.py \
  tests/test_extractions.py tests/test_feature_packs.py \
  tests/test_api.py tests/test_config_cli.py

# Existing ML interpreter; explicitly import development source
PYTHONPATH=/home/yc_liu/projects/HistoPilot-dev \
  /home/yc_liu/projects/HistoPilot/.venv-training/bin/python -m pytest -q \
  tests/test_mil_data.py tests/test_mil_model.py tests/test_mil_training.py \
  tests/test_refit_training.py tests/test_inference_execution.py tests/test_attention_execution.py

# Frontend and lint
export PATH=/home/yc_liu/.nvm/versions/node/v22.23.2/bin:$PATH
npm --prefix web test
npm --prefix web run build
.venv/bin/ruff check .
git diff --check

# Offline browser fixture: no web server
node scripts/build_review_ui.mjs
python scripts/verify_review_ui.py --browser /path/to/agent-browser

# Package the already built frontend
.venv/bin/python scripts/bundle_web.py
python -c 'import setuptools.build_meta as build; print(build.build_wheel("dist"))'
```

Offline `uv build` could not resolve uncached setuptools. Packaging therefore used the already installed setuptools 80.9.0 through its PEP 517 build backend; no installation or network download was necessary. The resulting wheel is `dist/histopilot-0.1.0.dev0-py3-none-any.whl` (a local ignored build artifact, not committed).

Full review logs are in `/tmp/histopilot-dev-review/`: `backend-tests.log`, `backend-final-changes.log`, `training-tests.log`, `ui-tests.log`, `build.log`, `ruff.log`, `browser-checks.log`, `wheel-build-local.log` and `wheel-verification.log`. These temporary logs supplement the committed outcomes and [browser check record](browser/checks.json).

## Browser evidence

The harness in `web/verification/devReview.tsx` loads the actual App shell with invented empty-project responses. The inference fixture imports the real production inference controls and validation helper. The compute fixture uses the real API client and job controls, simulates accepting a launch and losing its response, then verifies that an explicit retry reuses the same operation ID. No network request reaches a real API and no process launch is requested from HistoPilot.

The checks cover nine roadmap modules, the initial dataset action, matched fixture requests, horizontal overflow at desktop/mobile widths, mobile navigation with Escape/focus restoration, blank-versus-zero workers, partial decimal/exponent handling, bounds and hidden-field recovery, incompatible legacy aggregation, duplicate-click suppression, uncertain-launch retry identity, an injected render error, remount recovery and retained project navigation.

- [Desktop roadmap](browser/roadmap-desktop.png)
- [Mobile roadmap](browser/roadmap-mobile.png)
- [Inference controls](browser/inference-controls.png)
- [Compute retry](browser/compute-retry.png)
- [Render recovery](browser/render-recovery.png)

These screenshots are offline fixtures, not observations of the user's live project or GPU workload. The intentionally injected render error is expected evidence, not an unexplained browser failure.

## Boundaries

No HistoPilot server was started/restarted, no real research project was imported or changed, and no real training, extraction or evaluation workload was submitted. CPU regression tests do perform tiny synthetic model fits. Browser automation was closed after verification. `main` was checked and remained clean.

Live service-to-worker workflows with representative data, GPU/device faults, reboot recovery, storage failure and complete keyboard/screen-reader review remain unverified. See [remaining priorities](../DEV_REVIEW.md#remaining-priorities).

For manual trials, a Git branch isolates source code, not saved project data. Use a separate application workspace and new disposable project folders. For example, the user can start the already bundled dev UI on another available port:

```bash
cd /home/yc_liu/projects/HistoPilot-dev
HISTOPILOT_TRAINING_PYTHON=/home/yc_liu/projects/HistoPilot/.venv-training/bin/python \
  uv run histopilot serve --workspace "$PWD/.local/review-workspace" --port 8788 --no-browser
```

Open `http://127.0.0.1:8788` for the bundled UI. Add explicit source data roots as needed. Opening an existing research project still operates on that project's own folder; a separate application workspace does not copy or isolate it. The command above was documented, not executed.
