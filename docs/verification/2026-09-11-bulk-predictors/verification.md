# Bulk predictors and cohort evaluation verification

Implemented on 2026-09-11. HistoPilot was not started or restarted. Browser verification used a standalone local file with actual React components, synthetic records and mocked service methods; all network access from the fixture was blocked. No real project records, checkpoints or compute jobs were changed.

## Automated checks

- 273 frontend tests across 41 files passed; TypeScript passed.
- 100 focused backend regressions passed: predictor builds and identity, refits, experiment ownership, bulk evaluations, execution service, cleanup dependencies and API boundaries.
- The API routing test passed again after adding bulk-route authentication and request-validation checks.
- Ruff and Git whitespace checks passed.
- Production build passed and was bundled into `histopilot/static`. Vite reports the existing large-chunk advisory.

## Browser workflows

1. Selected two completed training seeds in the same experiment and configuration. Both at P75 reviewed four outputs, with a separate six-epoch refit budget for each seed.
2. Simulated failure of one refit-plan publication. The UI displayed its structured error. Retry submitted the identical operation and request, reused the three saved outputs, and completed the fourth without duplication.
3. Selected both refit plans, submitted two independent training requests, completed synthetic jobs and published both. The library retained each seed's ensemble and refit separately.
4. Reviewed all five predictors against one test cohort. Four were compatible; one was explicitly excluded. Added another predictor after review and simulated a lost submission response. Retry retained the same operation and fixed predictor snapshot, created only one evaluation batch and omitted the newly added predictor.
5. Cancelled the queued batch. All four pending evaluations became cancelled. Backend concurrency regression separately verified cancellation can interrupt submission between members.
6. Repeated a clean final build/evaluation flow. Four independent completed evaluation rows displayed their own method, seeds, cohort, accuracy and AUROC.
7. Verified the cohort deep link, result tables, desktop layout (1440px), and narrow layouts (600px and 390px). Document width equalled viewport width. Wide tables scroll inside their containers. Final browser error list and attempted network list were empty.

## Screenshots

- [Build predictors](build-desktop.png)
- [Per-seed predictor library](library-desktop.png)
- [Evaluation comparison](evaluations-desktop.png)
- [Evaluation at 390px](evaluations-mobile.png)

## Operational behavior

Both publishes ensembles and saves refit plans. Select plans under Refit jobs to train and publish them together. Training uses the existing shared resource reservations and isolated worker runtime.

Build reviews support 128 seed groups (256 method outputs); evaluation batches support 256 predictors. Larger workspaces can select subsets for successive batches. Active existing identities are reused. Archived or trashed identities must be restored. Retry uses its original reviewed inputs and does not restart failed compute attempts automatically.

Source inputs, frozen records and completed results remain protected by the existing archive/Trash dependency rules. Batch cancellation stops remaining submissions, requests cancellation of active children and retains finished results.
