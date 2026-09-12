# Unified stage workflows

The local scientific workspaces open at Stage 0: a saved-record library with an explicit create action. Selecting a record opens its own workspace. Model interpretation keeps its existing interface.

Experiments is the common opening-page template: one page title, a short purpose statement, and one primary create action, followed by a compact search/filter/sort toolbar and the saved-record table. The library does not repeat the page heading or show a Stage 0 banner. Names open records; Manage opens a page within the current stage, using the existing Archive, Delete and Restore controls with the exact record identity. Users do not navigate to Workspace cleanup. The underlying library stays mounted, preserving its search, filters and selections. Management stays open after a record becomes inactive, and Back is locked during confirmation or an uncertain response. Each phase keeps filters relevant to its records (draft/frozen, input readiness, lifecycle state, or evaluation method). Clear filters restores the default view, and empty search results remain distinct from an empty library. Experiment comparison controls appear when records are selected; evaluation results and batches occupy separate library tabs.

| Workspace | Pages after the library |
| --- | --- |
| Datasets | Choose files → map columns → review and freeze; existing datasets open for exploration or revision |
| Targets & splits | Development data → prediction target → split design → review and freeze |
| Slide features | Choose source → coverage → validation and optional packs → bundle review and freeze |
| Experiments | Details → inputs → batches → submission review → runs → results |
| Test cohorts | Test data → prediction target → review and freeze |
| Model evaluations | Experiments → evaluation inputs → compatibility review → batch results; single-predictor plans retain their separate input/review flow |
| Clinical utility | Evaluation evidence and settings → analysis review → saved report |

Batch configuration also uses separate configuration, training, compute/predictor, and review pages. Extraction and packing separate their settings, job review, and activity. Evaluation method comparisons have their own page.

`StageWorkflow.tsx` owns the shared library, step navigation, and focus/scroll transition. Scientific validation and publication remain in the existing workspace components and API clients. Moving between steps retains local form state; libraries provide resume actions for ongoing preparation. Submission, freeze, and uncertain-response retry controls retain their operation identities and locks. Clicking the current sidebar module opens its library without replacing deep-link parameters or remounting a pending publication.

## Verification

From `web/`:

```bash
npm test
npm run build
node scripts/verify-preparation-pages.mjs
node scripts/verify-feature-workflow.mjs
node scripts/verify-experiment-pages.mjs
node scripts/verify-test-cohort-workflow.mjs
node scripts/verify-evaluation-stage-navigation.mjs
node scripts/verify-record-management.mjs
node scripts/verify-system-compute.mjs
```

The browser scripts bundle the actual React components into temporary `file://` fixtures and drive local Chromium through a debugging pipe. Scientific APIs are mocked in memory: these checks start no HistoPilot server and launch no compute jobs. Each script prints its screenshot and verification-report directory. Coverage includes exact record identity, forward validation gates, state retained on Back/resume, freeze and submission, retry identities, locked records, and narrow-screen layouts. They do not replace live backend integration checks.

After building, `python scripts/bundle_web.py` from the repository root updates the Python package's generated UI assets. Server startup remains manual.

System & Storage polls the authenticated `/api/v1/system/compute` endpoint every five seconds, with pause and manual refresh controls. A cached sampler reads CPU counters, RAM/swap, workspace/data volume capacity, and NVIDIA utilization, VRAM, temperature and power when supported. It does not import a model runtime. The first CPU sample and unsupported measurements are explicitly unavailable; stale readings remain labeled. An already-running service needs a manual restart to load this new Python endpoint.
