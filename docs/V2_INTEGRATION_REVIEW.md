# HistoPilot v2 integration review

Reviewed 2026-09-11 against main's working tree and `hp_v2` commit `1391153`.

## Verdict

Bring over the incremental improvements, with corrections. Do not treat v2 as a completed redesign or merge its metric change blindly. Main already contained almost all the training, evaluation, experiment registry, and cleanup functionality visible in v2. The main product challenge remains the number of concepts and transitions users must understand; the backend's evidence checks and historical readers should survive any simplification.

The reviewed changes have been applied to the **main working tree**. Existing uncommitted work is preserved. No commit, branch fast-forward, remote push, server start, or server restart was performed. Saved projects and research data were not migrated or modified.

## What v2 actually changed

The `d6cbe7b` commit on `hp_v2` is a snapshot of main's preexisting uncommitted work. Every file in that snapshot matched main byte for byte at the start of this review. The meaningful v2 comparison is therefore **`d6cbe7b..1391153`**, not the much larger `main..hp_v2` Git diff. That delta covers 25 files, with 499 insertions and 39 deletions; 272 inserted lines are the design document.

| Area | Implemented on v2 | Integration decision |
| --- | --- | --- |
| Startup | Delay `h5py` import until packed-feature operations need it | Included; keeps HDF5 out of control-service startup |
| Refit evidence | Exclude experiment display names from freshness hashes | Included with repaired compatibility for genuine pre-v2 plans |
| Test cohorts | Exclude warning prose and presentation flags from preview identity | Included with repaired compatibility for genuine pre-v2 cohorts |
| Metrics | Average macro-F1 only over classes appearing in ground truth | Rejected; preserves main's fixed-class metric definition |
| Navigation | Dataset → protocol → features → experiments after saving | Included with exact input context and visible save confirmation |
| Terminology | Human-readable task/unit/split labels; more consistent protocol/bundle names | Included; broader terminology cleanup remains |
| Project links | Prefer `?project=` while reading old `?experiment=` links | Included; experiment IDs inside hash routes remain distinct |
| Local tools | Hide empty slide explorer/provenance sidebar links | Included; historical routes remain readable |
| Large redesign | Fewer modules, automatic version numbers, common UI primitives, event notifications, unified evaluation | Proposal only; not represented as shipped functionality |

## Corrections made during integration

### Scientific identity and compatibility

The original v2 compatibility tests replaced an outer hash on a new-format record. They did not cover both old refit hashes together or an older cohort with a real warning whose text subsequently changes. Old records could therefore still become stale after harmless edits.

The integrated logic authenticates the stored evidence using its original hash format before comparing current scientific content. Historical experiment names and name-dependent refit provenance remain preserved. Changes to checkpoints, recipes, membership, bindings, or blocking safety findings still prevent launch/publication. Tests exercise real old-format records, renamed experiments, reworded warnings, and corrupt or changed evidence.

See `application/predictors.py::evidence_current` and `application/evaluations.py::preview_current` under `histopilot/`.

### Metric semantics

Dropping every class with zero ground-truth support also drops a class that receives false-positive predictions. For true labels `[A, A]` and predictions `[A, B]`, v2 returns macro-F1 **2/3**, whereas main's fixed two-class average is **1/3**. The predicted-only class has defined F1 of zero and should not silently disappear from an established metric.

Main retains averaging over the declared class vocabulary, including its existing zero-denominator convention. This also preserves comparison with previously saved results. Regression tests cover absent classes, predicted-only classes, multiclass targets, patient aggregation, thresholded decisions, and unlabeled inference. A future alternative averaging convention should have an explicit name/version and reported support.

### Preparation and evaluation navigation

v2's redirects discarded source-page success messages when the page unmounted. A protocol saved against an older dataset could also open feature preparation using the project's newest dataset. The integrated routes carry dataset, protocol, and bundle identities; destination pages retain that context and display a save notice. Feature attachment/extraction and bundle selection use the selected dataset. Existing experiment inputs remain authoritative; compatible suggestions apply to new, unconfigured experiments.

Evaluation now honors both predictor and cohort deep links. Missing datasets and incompatible linked cohorts display an explicit unavailable selection rather than visually suggesting that a different option is selected. Tests cover the route context, multiple datasets, ambiguous compatible pairs, existing inputs, and unavailable records.

### Editing and recovery

- **Experiment metadata edits:** omitted inputs, notes, and tags remain saved; explicit null/empty values still clear them. Nested input defaults remain fully serialized. The API still requires the experiment name and reviewed revision.
- **Evaluation resume:** CPU/GPU resources are taken from the authenticated first execution plan. Newly available CUDA does not switch a CPU job to GPU. A missing originally selected GPU produces an actionable unavailable-device error; restoring that GPU allows the same plan to resume. Idempotent launch retries preserve their original request representation.
- **Training assessment resume:** an atomic `fit-complete.json` receipt binds the full execution plan, normalized recipe, output directory, history, and hashes of the best/last checkpoints. If prediction or metric writing is interrupted after that receipt exists, resume verifies it and repeats assessment without optimizing another epoch. Altered receipts, plans, paths, or checkpoints are rejected. Legacy runs without a receipt retain checkpoint-based resume; a crash before receipt publication and already archived old workers retain the older behavior.
- **Executable batch planning:** new batches reject unsupported v4 split modes during preview, before saving a plan that training would refuse. K-fold execution and historical protocol inspection remain available.

## UI verdict

The current interface has useful foundations: explicit required inputs, staged setup, readable coverage summaries, collapsible advanced controls, review-before-publication, retained dialog values on conflicts, and direct recovery links. The inspected desktop target setup and mobile feature review are legible, and the feature workflow has an understandable next action.

The remaining usability work is substantial but does not require discarding the backend:

1. **Standardize vocabulary across the whole journey.** “Targets & splits,” “development protocol,” “study design,” and “cohort” still coexist; training/evaluation screens also use “predictor,” “model,” “batch,” and “configuration.” Keep one primary label per object and explain technical terms in context.
2. **Reduce repeated publication interactions.** The source-save action now explicitly says “Save source & prepare bundle,” but sources, bundles, and other records still need separate publication steps. Preserve scientific checks while reducing repeated naming and explaining which object is being saved.
3. **Improve evaluation setup recovery.** A generic Test cohorts link exists, so the original “no fix path” finding is overstated. It should carry a predictor's exact development protocol and bundle into test setup and explain why a cohort is incompatible.
4. **Unify tab/stepper behavior.** Multiple implementations have differing styles and keyboard conventions; some tablists lack Arrow/Home/End navigation. Current buttons remain keyboard activatable, but that is not complete tablist behavior.
5. **Load less code and avoid redundant refresh work.** The production build still reports a large entry chunk, approximately 816 kB minified / 216 kB gzipped. Route-level loading is a concrete next improvement. Polling also overlaps across job/registry surfaces.

The large navigation/module consolidation in `HP_V2_DESIGN.md` remains a design option, not a prerequisite for adopting these fixes. Demo removal, required-tag removal, and replacing the roadmap should be deliberate product decisions supported by workflow tests.

## Backend verdict and remaining limits

The code protects important boundaries: immutable scientific references, journaled publication, patient grouping, validation-only checkpoint selection, exact development assessment membership, checkpoint hashes, development/test overlap checks, isolated compute workers, and process-aware job ownership. Targeted tests cover these contracts; this review is not a performance benchmark or validation of a real research dataset/model.

| Priority | Remaining issue or decision | Evidence and implication |
| --- | --- | --- |
| High for study validity | External label meaning and test-driven model selection | Same-dataset/same-field label inversions now block, and external remapping of shared raw values warns. The software cannot infer that two external label vocabularies mean the same thing. Primary-model and operating-threshold preregistration remain unenforced. |
| Medium | Read-time verification scales with data size | Scientific reads repeatedly initialize/check storage; protocol dataset loading repeats artifact hashing; feature resolution stats source files. Polling can multiply this work. Measure realistic projects before adding bounded caches; retain launch/publication verification. |
| Medium | Lifecycle history has a finite growth limit | `storage/lifecycle.py` replays retained audit/operation history and caps serialization at 16 MiB. No compaction is implemented. A sufficiently long-lived project can stop accepting lifecycle mutations; existing valid state remains readable. |
| Medium | Archived records have different new-selection rules | Bulk evaluation excludes archived predictors while single evaluation can retain one. Archived provenance is intentionally valid. Define a consistent new-selection policy without invalidating existing historical references. |
| Low, qualified | Abandoned publication reservations need an explicit recovery policy | Recovery defensively skips trashed sources while retaining a preparing publication's tag reservation; restoring sources permits recovery. A normal UI path creating that combination was not confirmed, because cleanup attempts recovery under the same lifecycle lock before trashing. This is not a reproduced permanent leak. |
| Medium | Limited interaction/performance coverage | Most committed UI tests use static markup or pure functions. Offline browser checks complement them, but do not exercise real service requests, GPU behavior, large inventories, or a complete real-data study. |

Only v4 k-fold is currently executable by native training/predictor workers. New training batches now reject unsupported designs before publication. Old protocol readers and supported development-pool selection are preserved: `development_splits.py` generates memberships used by actual k-fold execution and is not disposable preview-only code.

The original document also understates existing provenance: runtime records already include Python/platform/CUDA information, and compute archives authenticate all archived Python files. The lighter execution fingerprint does not include every dependency, which is narrower than saying ordinary resumes silently use changed schema code.

## Verification

Tests use temporary fixture projects; browser checks use an offline file with synthetic responses. The HistoPilot service was neither started nor restarted.

- UI: 282 tests passed across 43 files; TypeScript passed; production Vite build passed with the entry-size warning described above.
- Optional training runtime: 66 MIL/refit/inference tests passed, including multiprocessing and interrupted assessment; run with `.venv-training/bin/python`.
- Full backend suite: **1,249 passed, 14 skipped**, with two dependency deprecation warnings, in 481 seconds. The separate training-runtime run above covers optional ML tests. Log: `/tmp/histopilot-v2-integration/backend-tests.log`.
- Additional batch support tests: 39 tests passed, including three unsupported-mode cases added during the broader suite run.
- Ruff and whitespace checks passed. CI now runs Ruff and UI tests as well as backend tests and the production build.
- The production frontend was copied into `histopilot/static` with `scripts/bundle_web.py`, ready for the next user-started local service. An installed wheel elsewhere still needs its usual rebuild/reinstallation.
- Offline browser: target setup and feature coverage review inspected at desktop/mobile sizes; no browser errors, horizontal overflow, or attempted network requests in inspected states. A protocol link retained its older dataset and displayed the save notice despite a newer default dataset. This does not certify a live service integration or real GPU study.

API TestClient and multiprocessing checks stalled inside the restricted execution sandbox; the affected runs were stopped and rerun successfully outside that sandbox. No application change was made to conceal that environment issue.
