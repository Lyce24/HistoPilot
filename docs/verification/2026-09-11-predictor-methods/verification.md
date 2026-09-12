# Predictor methods and naming verification

Implemented the revised module names, refit and ensemble predictors, durable refit/evaluation jobs, result downloads, and long-text layout fixes. Existing route IDs and scientific records retain their identities.

## Automated checks

- 267 frontend tests in 40 files passed; TypeScript passed.
- 31 cleanup and experiment regression tests passed, including refit/evaluation dependency protection, cancellation replay, and deleting an unlaunched plan.
- 29 predictor/refit service regressions and the authenticated predictor/refit API test passed.
- 45 synthetic CPU refit/MIL training tests passed, including bitwise comparison of resumed and uninterrupted refit weights. An additional test then exercised the pinned worker subprocess from a saved refit plan through verified predictor publication.
- 10 compute/evaluation service tests and 6 actual CPU inference tests passed. These cover pinned source execution, probability rather than logit averaging, positive-class threshold order, unlabeled records, stable finite losses, and receipt validation.
- Ruff and `git diff --check` passed for the integration.
- Production Vite build passed and four frontend files were bundled into `histopilot/static`. The existing bundle-size advisory remains nonfatal.

## Offline browser verification

The actual React pages were built into an isolated file-based browser harness. API methods returned synthetic data in memory; fetch was blocked and monitored. No HistoPilot server was started or restarted, no real training or inference job was launched, and no actual project record was modified. The browser session was closed afterward.

Verified interactions:

1. Experiment list with production-shaped legacy IDs and long protocol/feature IDs: no overflowing cell text at 1440px or 600px. Document width equals viewport width; the table scrolls inside its container. Selects reserve 36px for their arrow. Full identifiers remain in titles and record details.
2. Predictor choices expose fold ensemble and full-development refit. The library displays both methods for one experiment and an independent second experiment.
3. Custom percentile 101 blocks preview without making a request. Entering 75 succeeds and preserves it in the reviewed selection. The review shows the best epochs and derived six-epoch budget for `[2, 3, 5, 8]`.
4. Refit requires a saved plan before Train appears. Launch, Cancel, Resume and completed publication are separate actions. Publish appears after a completed receipt; the published predictor opens evaluation with the correct ID.
5. Refit evaluation restricts the cohort picker to matching development protocol and feature provenance. Stale cohorts are disabled. Saving a plan does not launch inference; Run evaluation is a separate action.
6. Completed worker results render metrics and prediction downloads. Slide prediction download carries the correct evaluation ID. Unavailable patient metrics display an explanation rather than fabricated scores. The 600px evaluation layout stays inside the viewport.
7. Zero browser runtime errors and zero fetch/network API calls were observed.

The final screenshot fixtures seed completed records so the library can show both methods together after rebuilding the latest code. Screenshots cover the predictor library, long experiment identifiers, narrow experiment layout, and evaluation results. Backend tests independently exercise real temporary CPU model execution; the browser checks use mocks and are not a live-server end-to-end test.

Artifacts: `predictor-methods.png`, `experiments-long-ids.png`, `experiments-mobile.png`, `evaluation-mobile.png`, `harness.tsx`, `build.mjs`. Harness build paths document this workstation's temporary verification environment and are not part of the shipped application.
