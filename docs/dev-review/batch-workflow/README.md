# Batch planning and experiment evaluation redesign

The earlier design stored one predictor policy for an entire experiment. That could not represent batches with different intended outputs or refit budgets. The revised flow saves predictors with each batch, verifies shared inputs before batch editing, and makes experiment selection the starting point for evaluation.

## Implemented behavior

- Planning follows **Inputs → Batches**. Select protocol and bundle, then **Check & continue to batches**. Unsaved input changes prevent editing batches against a different saved pair; pending batch changes prevent shared-input edits.
- Each batch editor orders template, name, parameter search/repeats, seeds, settings, compute/parallelism and predictor configuration before one Add/Save action. Batch cards show their predictor method and expected count. Importing batch settings retains the experiment’s verified inputs; copying a whole experiment still copies its inputs too. Multiple batches can choose different methods and percentiles.
- New standard recipes use **40 maximum epochs / early-stopping patience 8**. The quick-check template explicitly uses 5/3. Existing recipes retain saved values; historical omitted budgets retain their former 100/15 interpretation.
- New submission receipts pin a versioned policy map by frozen batch ID. Each refit job carries its own percentile. Copying earlier experiments resolves inherited policies into editable batch settings. Historical submitted global-policy workers retain their original archive and intent.
- Evaluation starts with explicit selection of experiments, their ready methods and a test cohort. No experiments selected means no jobs. Review fixes exact predictor IDs; later arrivals need another review. Setup navigation preserves pending operation identities.
- Results compare ensemble/refit source pairs within the same cohort, scoring unit, class encoding, aggregation, threshold and labeled count. Means use the same finite paired observations. Missing pairs stay visible; the interface does not choose a winner from unmatched scores.
- Review also closed an older bulk predictor API bypass after cancellation. Missing or inconsistent frozen predictor intent fails closed, with an actionable status rather than silently becoming a legacy experiment.

## Validation

- Frontend: **434 tests passed** in 66 files; TypeScript and production build passed.
- Final backend policy/recovery regressions: **82 passed**, after the cancellation and corrupted-intent fixes.
- Synthetic CPU refit tests: **5 passed**.
- Offline browsers: **89 checks passed** (38 experiment lifecycle/input handoff, 29 batch editor/template ownership, 22 evaluation selection/comparison). No page errors or page overflow at desktop/mobile widths.
- Ruff and whitespace checks passed.
- Bundled the frontend and rebuilt the wheel; all 145 Python files and four frontend files match the current source/build, with no stale frontend assets. See [package verification](package-check.json).
- The existing Vite bundle-size warning remains: the single entry is 977.45 kB (259.82 kB gzip).
- Full backend suite: **1,569 passed, 16 skipped**, in 900.20 seconds. The skips require optional dependencies in the lightweight server environment. This suite started before the final bulk/corrupted-intent guards; the affected 82-test suite was rerun afterward against the final code. See [full suite log](backend-tests.txt) and [final guard regressions](backend-policy-tests.txt).

Browser fixtures use invented records and mocked API replies; they exercise the actual React components without opening a server or submitting research jobs. Synthetic CPU refit tests use temporary data.

Reproduce from the repository with Node 22 on PATH:

```bash
.venv/bin/python -m pytest -q
.venv-training/bin/python -m pytest -q tests/test_refit_training.py
(cd web && npm test && npm run build)
.venv/bin/python scripts/bundle_web.py
node scripts/build_experiment_review.mjs
.venv/bin/python scripts/verify_experiment_review.py --browser /path/to/agent-browser
node scripts/build_batch_review.mjs
.venv/bin/python scripts/verify_batch_review.py --browser /path/to/agent-browser
node scripts/build_evaluation_review.mjs
.venv/bin/python scripts/verify_evaluation_review.py --browser /path/to/agent-browser
```

The user starts or restarts the HistoPilot server manually. This review starts no app server and changes no project data. Refit execution remains sequential within an experiment; each evaluation batch accepts at most 256 predictors. The supported training path remains k-fold ABMIL classification.

## Screenshots

- [Protocol and feature bundle inputs](experiments-inputs.png)
- [Multiple batch plans](experiments-planning.png)
- [Ordered batch editor](batch-editor-desktop.png)
- [Batch predictor settings](batch-predictor-settings.png)
- [Visible choices across batch cards](batch-plan-predictor-choices.png)
- [Mobile predictor settings](batch-predictors-mobile.png)
- [Refit tracking](experiments-refit-tracking.png)
- [Experiment selection for evaluation](../evaluation/evaluation-experiment-predictors.png)
- [Reviewed predictor snapshot](../evaluation/evaluation-reviewed-selection.png)
- [Paired method comparison](../evaluation/evaluation-method-comparison.png)
- [Evaluation on mobile](../evaluation/evaluation-mobile.png)
