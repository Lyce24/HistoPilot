# Experiments redesign verification

Branch: `HistoPilot-dev`. Verified 2026-09-12 using temporary scientific projects, mocked executors and invented browser records. No HistoPilot server or real training job was started or restarted. The main checkout remains unchanged.

## Results

| Check | Result |
| --- | --- |
| Full backend regression suite | 1,514 passed, 16 skipped, 2 dependency deprecation warnings; 739.60 seconds |
| Final affected backend suites | 223 passed, 9 skipped, 2 dependency deprecation warnings; 338.13 seconds |
| Full frontend suite | 396 passed across 56 test files |
| TypeScript and production UI build | Passed |
| Offline real-browser workflow | 26 checks passed; see [checks.json](checks.json) |
| Python lint, changed-file formatting and diff whitespace | Passed |
| Bundled UI and wheel packaging | Passed; all 145 packaged Python/static files match the source checkout |

The full suite ran during final integration; the subsequent focused backend run covers the last submission environment contract, copy deduplication and frozen-input consistency changes. It includes experiment stages, experiment records/API, training execution, development batches, epoch history, workspace cleanup, lifecycle storage and job lifecycle. Optional native Torch tests are skipped in the lightweight control environment. Execution tests use fake executors; browser submission requests are intercepted by the fixture. No actual GPU training outcome is claimed.

The production build retains the existing large-bundle warning (about 931 kB minified / 247 kB gzip for entry JavaScript). This change does not redesign code splitting or the existing process-descendant lease recovery behavior.

## Coverage

- Planning-only input and recipe mutation; revision conflict protection, including remote changes to a clean open editor.
- Complete source experiment copies, archived sources, equivalent historical recipe deduplication, and isolated new ownership without copied runs/results.
- Scientific/runtime preflight before locking; durable submission receipts; partial publication, failed worker start and lost acknowledgement recovery without duplicate launches.
- Common worker code and dependency environment across a submission; explicit recovery after environment drift.
- Typed and generic API locks, retained history through lifecycle changes, no reopening by hiding a batch, and no resuming a finished experiment.
- Run membership, bounded epoch metadata, invalid/nonfinite history, safe file reads, history refresh on terminal transitions and incomplete-score exclusion.
- Creation and templates, multi-batch editing, pending submission locks, same-operation retries, readable immutable inputs/batches, selected-run loss curves and GPU samples, concise finished results, stable tabs under polling and mobile overflow.

## Reproduce

Use the checkout's Python environment and a supported Node version (the checks used Node 22.23.2):

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
cd web
npm test
npm run build
```

From the repository root, build and test the standalone browser fixture:

```bash
node scripts/build_experiment_review.mjs
python scripts/verify_experiment_review.py --browser /path/to/agent-browser
```

This builds a `file://` document under `/tmp/histopilot-experiment-review/browser` and mocks every API request. It opens no server port. The local verification logs are under `/tmp/histopilot-experiment-review/`.

## Screenshots

- [Experiment list](experiments-list.png)
- [Editable planning](experiments-planning.png)
- [Running experiment overview](experiments-running.png)
- [Run table and resource usage](experiments-tracking.png)
- [Saved loss history](experiments-loss-history.png)
- [Finished results](experiments-results.png)
- [Mobile view](experiments-mobile.png)
