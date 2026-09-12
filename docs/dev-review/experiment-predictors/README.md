# Experiment-owned predictors review

Reviewed on `HistoPilot-dev`, September 12, 2026. This extends the earlier [experiment lifecycle review](../experiments/README.md) with predictor ownership and a direct Experiments → Evaluate models flow.

The earlier separation required users to reconstruct configuration/seed groups after training and manually launch and publish each refit. The new experiment plan saves **Skip / Refit / Ensemble / Both**, applies it consistently across every batch, and permanently freezes it at submission. The expected count is based on complete k-fold groups: **3 training seeds × 15 configurations × 2 methods = 90 predictors**, from 225 individual five-fold training runs.

Refit budgets use the selected percentile of best-checkpoint epochs with linear interpolation and rounding up. Refit sees the complete development cohort without test data or early stopping. A persistent coordinator creates verified ensembles and runs/publishes refits automatically. It waits for a source batch to finish before publishing its checkpoint evidence, serializes refits within an experiment, and uses existing shared compute reservations. Scientific inputs, runtime and archived worker code remain fixed across retries.

Independent review caught and corrected cleanup's missing filesystem dependency, experiment status incorrectly saying Finished while predictors were pending, and operation retries that could affect a later coordinator attempt. Cleanup checks live coordinator ownership even after a terminal receipt. Status polling never dispatches predictor work, and historical submissions without an integrated policy never acquire new jobs.

Final review also reproduced an older manual refit API reopening cancelled predictor work. Creation, launch and publication now enforce the closed experiment intent. Existing predictor evidence remains readable for evaluation, and accepted publication retries retain their original receipts.

The batch editor separates search mode, training defaults, optimization, model architecture, precision and resources. Custom configurations open individually. Partial numeric text remains editable but cannot be submitted as an invalid value; hidden invalid fields open for correction. Duplicate scientific configurations do not inflate counts. The evaluation page groups ready outputs by source experiment, configuration, training/split seeds and method. A review freezes the exact filtered predictor IDs.

## Final verification

| Check | Result |
| --- | --- |
| Full backend integration suite | 1,558 passed, 16 optional tests skipped; 801.38 seconds |
| Final predictor orchestration and closed-intent guards | 36 passed; 44.73 seconds |
| Related predictor registry/build, refit and experiment lifecycle cases | 70 passed with the final guards |
| Final native CPU refit checks | 5 passed; fixed epoch budget, no test/validation loaders, exact interruption/resume, archived worker execution and verified publication |
| Full frontend suite | 415 passed across 62 files |
| Offline browser workflows | 74 checks passed across experiments, batch editing and evaluation |
| TypeScript, production build, Python lint/format, diff and documentation links | Passed |
| Packaged wheel | All 149 Python, resource and static files match the working source; no stale static assets |

The full backend run preceded the final closed-intent guard; the focused runs above cover that last change and its related legacy paths. The native tests use tiny temporary synthetic data and CPU execution. Optional Torch tests are skipped in the lightweight control environment; the separate training-environment run exercises real refit execution. Warnings are existing dependency deprecations and expected small-test DataLoader notices. The production entry bundle is 966.37 kB (255.98 kB gzip).

Wheel: `dist/histopilot-0.1.0.dev0-py3-none-any.whl`, SHA-256 `0da529ec81429e31abc04791934ced5ea7e2808c3110e0898cf35384eff34d7d`.

## Browser verification

- **41 experiment checks**: template copy including predictor policy; all four choices; invalid percentile rejection; permanent locks; lost submission and predictor-action responses; refit progress after folds finish; finished predictor library and evaluation handoff; desktop and 390px mobile layout.
- **16 batch checks**: organized settings; 15 × 3 training groups; invalid/duplicate seeds; grid-to-custom inheritance; duplicate configuration deduplication; hidden invalid fields; zero dropout; preserving edits; read-only submission state; desktop/mobile layout.
- **17 evaluation checks**: 90 generated predictors; filtered 45-refit review/submission; exact source isolation; Skip and missing-source states; source navigation; desktop/mobile layout.

These fixtures render the real React components against invented records and mocked requests, entirely through local `file://` pages. They never contact the HistoPilot server or launch research jobs. Scripts verify there are no unhandled requests or browser page errors. The mobile checks found and fixed intrinsic grid width expanding the predictor tables beyond the page.

Reproduce from the repository root with Node 22 on PATH:

```bash
node scripts/build_experiment_review.mjs
python scripts/verify_experiment_review.py --browser /path/to/agent-browser
python scripts/verify_batch_review.py --browser /path/to/agent-browser
node scripts/build_evaluation_review.mjs
python scripts/verify_evaluation_review.py --browser /path/to/agent-browser
```

The batch driver uses its own output directory; build that fixture with `node scripts/build_experiment_review.mjs /tmp/histopilot-batch-review/browser` before running it. JSON check receipts are retained alongside these screenshots:

- [Predictor planning](experiments-planning.png)
- [Refit tracking](experiments-refit-tracking.png)
- [Finished predictors](experiments-results.png)
- [Mobile experiment](experiments-mobile.png)
- [Batch editor](batch-editor-desktop.png), [mobile batch editor](batch-editor-mobile.png)
- [Evaluation source selection](../evaluation/evaluation-experiment-predictors.png), [reviewed evaluation](../evaluation/evaluation-reviewed-selection.png), [mobile evaluation](../evaluation/evaluation-mobile.png)

## Boundaries

Verification uses temporary synthetic data. It does not establish performance on real cohorts or clinical validity. Automatic refits run sequentially within one experiment; other experiments still share the existing scheduler. Old predictor/refit records and legacy recovery links remain accessible. The app retains the existing large JavaScript entry-bundle warning. Server startup and restart remain manual.

To inspect the rebuilt branch, stop the previous dev server yourself and start it from the dev checkout, then open `http://127.0.0.1:8788`:

```bash
cd /home/yc_liu/projects/HistoPilot-dev
uv run histopilot serve \
  --workspace /home/yc_liu/projects/HistoPilot-dev/.local/review-workspace \
  --port 8788 --data-root /mnt/d --data-root /mnt/wsl/oceanpath-hot --no-browser
```
