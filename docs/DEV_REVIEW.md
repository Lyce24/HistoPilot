# HistoPilot development verdict

Reviewed 2026-09-12. Branch: **`HistoPilot-dev`**, created from **`722343e`** in `/home/yc_liu/projects/HistoPilot-dev`. The existing clean `dev` worktree was reused to create the requested branch. `main` and the separate `hp_v2` worktree were not modified.

## Verdict

**HistoPilot has a credible foundation for supervised, single-user pathology research. Its scientific persistence and separation of development from evaluation are stronger than its operational recovery and some of its UI guidance.** The most valuable improvements are to make its promises match what actually runs, catch incompatible inputs earlier, and keep failures understandable and recoverable.

The application already implements imports, grouped protocols, extraction, verified feature bundles, ABMIL k-fold training, predictor construction, test inference, clinical reports and attention. Old architecture and System descriptions understated those capabilities. Other controls overstated them: spreadsheet covariates could be selected although ABMIL ignored them, and maximum patient aggregation could be frozen although inference rejected it. Both kinds of misleading guidance matter.

This branch implements the concrete correctness and reliability fixes below. It does **not** establish clinical validity, multi-user production readiness, comprehensive hardware-failure recovery, or complete usability across every saved project. The remaining limitations are explicit rather than hidden behind a claim that everything is robust.

Supporting reviews: [backend/execution](dev-review/backend.md), [scientific workflow](dev-review/scientific-workflow.md), [UI/workflow](dev-review/ui.md). The updated [architecture](ARCHITECTURE.md) describes the implemented system.

## Findings and implemented changes

Priority reflects effects on scientific interpretation, resource ownership or completing a workflow; it is not a security vulnerability score.

| Priority | Original failure or misleading behavior | Change in this branch |
| --- | --- | --- |
| High | Protocol spreadsheet predictors were silently ignored by native ABMIL. | Block nonempty covariates during MIL input review. Explain image-only ABMIL and allow removing legacy selections. |
| High | Imports could rename the same WSI and bypass slide-ID overlap checks. | Compare recorded canonical paths and file identity stamps alongside slide/patient IDs; block known development-source aliases. |
| High | Inference lost patient-ID provenance; clinical reports could treat Slide_ID fallback groups as verified independent patients. | Retain provenance, recover omitted legacy provenance from frozen memberships, reject contradictions, block fallback patient analysis and suppress unjustified slide intervals. |
| High | PID/lease registration could fail after a worker started but before cleanup tracked it. | Record ownership immediately after spawn, close logs on spawn failure and drain tracked children on registration failure. Continue cleanup after exit-versus-signal races. |
| High | Lost tmux acknowledgements could overwrite active or completed worker evidence with launch failure. | Reconcile all four execution families. Read current worker state after session inspection, preserve worker-owned outcomes and explain uncertain acknowledgement. |
| High | Corrupt optional progress could disable status and cancellation; deeply nested metadata could fail HTTP encoding. | Validate optional progress fields and bound JSON nesting; return a warning while retaining controls. Required evidence remains strict. |
| High | Blank numeric inference inputs became zero, and button-driven saves skipped validity checks. | Strict numeric entry and validation before save/review/freeze; reveal invalid fields inside collapsed details. Blank/partial/out-of-range input blocks submission. |
| Medium | Unsupported patient aggregation and feature dtype mismatches failed late. | Detect them during cohort review; offer supported mean aggregation and explicitly show incompatible legacy choices. |
| Medium | Clinical eligibility errors were mislabeled malformed data; namespace warnings disappeared from analysis. | Preserve actionable storage errors and include independent-patient-namespace warnings in clinical reports. |
| Medium | Stopped cancelled compute looked interrupted; concurrent first-worker directory creation could fail. | Respect durable cancellation intent and safely accept concurrent creation of the same managed directory. |
| Medium | Late parallel 401 responses could discard renewed sessions; network errors lacked useful recovery guidance. | Share session renewal across JSON/images/downloads; preserve cancellation, error codes and invalid-field names; explain uncertain mutation outcomes without automatic replay. |
| Medium | A render exception could blank the application. | Project/module render boundaries provide retry, error details and navigation recovery, with an explicit unsaved-input caveat. |
| Medium | Partial job loading looked complete and missing status looked ready to run. | Distinguish loading/unavailable/stale states, retain known activity and expose retry. Scope uncertain actions to their exact project/job. |
| Medium | Computed outputs were called frozen versions; inaccessible unfinished work could imply roadmap completion. | Output-specific actions and an all-modules-complete check, otherwise prerequisite recovery. |
| Medium | System and architecture called existing training/storage capabilities unimplemented. | Separate native execution capability, tmux and TRIDENT readiness; correct storage/diagnostic descriptions and preserve the old architecture as history. |

## Backend and scientific assessment

**Preserve the evidence boundaries.** Content identities, frozen memberships, optimistic revisions, guarded writes, publication journals and staged artifacts are substantive safeguards. Tags and lifecycle visibility are separate from scientific contents. Source archives and runtime contracts constrain resume. Feature bundles track validation/freshness, and process exit alone is not proof of usable artifacts. Replacing these with a simpler-looking mutable project document would weaken reproducibility.

**Patient identity is a scientific input.** Grouped splits and patient aggregation depend on reliable mappings. Explicit slide fallback can support slide-oriented work but cannot establish patient independence. Exact IDs, source paths and file stamps detect some overlap; copied/transformed slides and independently named patients still need external identity reconciliation. No detected overlap is not proof of independent populations.

**Development results support model selection.** Saved partitions distinguish fitting, early stopping and development assessment; training seeds do not redraw splits. OOF comparisons do not create an independent final estimate after configuration selection. Test cohorts and frozen predictors are separate for that reason. Native execution currently supports k-fold ABMIL; other split designs must stay blocked until their execution/selection dependencies are implemented.

**Keep feature identity separate from loading convenience.** Packing is optional. Full tensor checks and current bundle evidence matter more than file size. Encoder names and dimensions alone do not authenticate checkpoint provenance. Precision conversion and source freshness checks must survive pipeline optimizations.

**Preserve evidence when failures occur.** Progress is not authoritative state. Missing HTTP/tmux acknowledgement does not prove rejection. Cancellation requires stopped-process evidence; resume requires compatible inputs, code and checkpoints. The changes improve these boundaries without deleting artifacts or migrating research records.

## UI and workflow assessment

The existing structure is worth keeping: a roadmap, explicit saved outputs, preparation steps, context-aware links, and separate experiment/predictor/test registries. It makes a complex research process inspectable. A broad visual redesign would add transition risk without addressing the most important defects found here.

The design criterion should be **one truthful next action with its reason**. Disabled actions should explain prerequisites; stale views should say their evidence is stale; save errors should preserve input; unsupported settings should be caught before an expensive run. Numeric, job-state, roadmap, System and covariate changes apply that criterion.

Labels, Radix dialogs, focus styling, skip links, responsive layouts and reduced-motion styles provide useful accessibility foundations. Selected controls and desktop/mobile overflow were checked in a real browser. This does not establish complete keyboard or screen-reader usability for every table, modal and viewer.

## Compatibility and existing work

- No scientific versions, names, project folders or user artifacts were migrated or rewritten. The HistoPilot server was not started or restarted.
- Valid historical cohort hash formats remain readable. Source-overlap details are added only when overlap is detected. Stricter preflight can correctly block settings that could not execute as advertised.
- Review historical experiments whose protocols contain spreadsheet covariates: earlier ABMIL runs did not consume them. Clearing a selection does not change an existing trained model.
- Saved clinical analyses retain their original immutable reports. Recreate affected analyses if patient identities came from Slide_ID fallback. Corrected review recovers provenance from frozen memberships when older prediction files omitted it.
- New control-response fields are additive. Legacy health/generic-jobs execution flags refer to the reserved generic executor, not project-specific training/evaluation.
- Optional telemetry corruption is recoverable; corrupt required plans, checkpoints or result evidence remain blocking failures.

## Remaining priorities

| Priority | Remaining work | Practical limit of this pass |
| --- | --- | --- |
| High | Consistent unsaved-change handling across Data, Protocol and Test-cohort editors, including save/discard/stay on navigation and unload | Drafts require explicit save. A partial guard would behave inconsistently around automatic post-freeze navigation and project switching. |
| High | Failure drills for surviving descendants, lost supervisors, ignored signals, resource reservations, full disks and reboot/resume | Controlled regressions do not prove every OS/driver/process-tree outcome. Group ownership needs careful PID-reuse handling and coordinated lease semantics. |
| High for external studies | Cross-dataset identity reconciliation beyond IDs and file aliases | Copied/transformed images and independent patient namespaces need stronger evidence. No automatic clinical-independence guarantee is made. |
| Medium | Extract cohesive editor steps and introduce route-level loading | Protocol exceeds 2,000 lines; Dataset exceeds 1,000. Entry JS remains about 909 kB minified / 242 kB gzip. A routing refactor needs dedicated interaction coverage. |
| Medium | Profile large inventories, job lists, tables and polling | Thousands of jobs and large WSI cohorts were not benchmarked. Preserve freshness and identity checks when optimizing. |
| Before broader deployment | Authentication/authorization, recovery procedures and realistic user trials | The supported boundary remains a single-user loopback workstation; this review is not clinical software validation. |

## Verification

The [consolidated verification record](dev-review/verification.md) contains results and reproduction commands. Tests use temporary projects, synthetic tensors, mocked executors and controlled failures. The offline browser harness builds actual development components with invented project responses; it does not contact a HistoPilot service.

Run backend tests with `.venv/bin/python -m pytest -q`, lint with `.venv/bin/ruff check .`, and frontend checks with `npm test` and `npm run build` from `web/` using a supported Node version. Build offline UI checks with `node scripts/build_review_ui.mjs`, then run `python scripts/verify_review_ui.py --browser /path/to/agent-browser`. These commands do not start a HistoPilot server. A live end-to-end trial must use a server started by the user.
