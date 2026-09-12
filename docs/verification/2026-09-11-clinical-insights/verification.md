# Clinical insights implementation review

## Verdict

The requested fourth stage and two modules are implemented in the local main working tree. The roadmap now contains nine modules across Prepare, Develop, Evaluate, and Clinical insights. Clinical utility consumes saved evaluation evidence; Model interpretation computes real native ABMIL attention for arbitrary compatible slides using refit or ensemble predictors. The source workflow is linked in both directions: experiment → predictor → evaluation → clinical analysis → attention study.

The implementation passes automated correctness and lifecycle checks and an offline browser review using the actual React components. Browser service responses were controlled fixtures; the attention image and map came from real inference with two distinct tiny CPU ABMIL checkpoints. No HistoPilot or Vite server was started. Clinical validity for a user's own cohort and real vendor WSI performance were not established by these synthetic tests.

## UI review

| Concern | Implemented behavior and verification |
| --- | --- |
| Navigation | Saved records carry their experiment, predictor and evaluation IDs. Returning to a predictor opens that exact predictor in its library. A clinical report forwards its predictor/evaluation/report IDs into interpretation. Archived linked reports and studies remain readable. |
| Clinical meaning | Analysis unit, positive outcome, labeled/unlabeled counts, frozen versus descriptive threshold, metric definitions, reference strategies and unavailable statistics are visible. Saved settings reopen accurately. |
| Curves | Seven charts cover calibration, ROC, precision–recall, operating characteristics, net benefit, clinical impact and net interventions avoided. Numeric tables are lazy and bounded. The focused net-benefit view explains its clipping and provides a full-range option. |
| Input clarity | Interpretation separates slide image, feature HDF5, coordinates, geometry and row-alignment confirmation. Optional links do not prevent standalone interpretation of compatible slides. Server file browsing and actionable review findings are provided. |
| Attention meaning | The UI explicitly labels class-independent pooling attention and percentile colors. The raw normalized weight remains inspectable. Ensemble mean and individual members are selectable. |
| Viewer correctness | Matching level-0 bounds are requested for image and attention after zoom and keyboard pan. Opacity endpoints change the rendered layer. Overlapping patches use the strongest visible attention consistently for color and inspection. |
| Responsive behavior | Desktop at 1440 pixels and mobile at 390 pixels showed no page-level horizontal overflow. All nine modules and four stages were present. The attention legend was corrected after detecting a collision with legacy demo styling. |
| Errors and incomplete work | Missing prerequisites, incompatible sources, partial overlays, pending jobs, cancellations and failed results are explicit. No fabricated heatmaps or completion states are substituted. No uncaught browser errors were observed. |

## Backend review

Clinical reports require a completed evaluation and checksum-verified prediction/metric artifacts. Frozen target encoding, cohort membership, patient aggregation and source references are cross-checked. Calculations preserve the selected analysis unit; unlabeled observations do not enter outcome-dependent denominators. Binary and one-versus-rest multiclass rules are explicit. Tied scores are grouped for exact AUC/AP. A Brier prevalence reference is descriptive, and no test threshold optimization or calibrator fitting occurs.

Independent numerical checks compared 100 randomized synthetic reports against scikit-learn for Brier score, ROC AUC, average precision and log loss at interior probabilities. Native saved log probabilities preserve extreme finite losses. Legacy zero-probability handling uses the documented floor. Wilson intervals are withheld from slide analyses with repeated or missing patient identities.

Interpretation verifies checkpoint identity, target, encoder, tensor dimensions/dtype, feature/coordinate row counts, finite tensors, geometry and source changes. It uses the complete slide bag and the exact frozen members, with no local softmax or silent patch sampling. Each ensemble member's normalized attention and class probabilities are computed before averaging. Resume receipts bind the input and artifact identities and preserve row ordering.

Image reads are bounded. Fractional source crop boxes avoid stretching rounded pyramid reads into a different geometry. Numeric sidecars use object-free read-only memory maps with checksum/stat validation, bounded paging and closed mappings. Full JSON exports use an atomic streaming writer with an explicit artifact budget. File requests and downloads remain authenticated and scoped to configured roots. Control API imports remain free of Torch, Lightning, HDF5 and imaging runtimes.

Cleanup follows the entire record chain. Retained clinical and interpretation records protect their inputs, including archived dependencies. Active attention jobs block cleanup until stopped, cancellation routes to the compute service, and restore checks dependent references. Scientific changes require a new review; changes to interpretation help text do not invalidate already saved scientific inputs.

## Verification

- Full backend regression: **1,319 passed, 15 skipped**, approximately 8 minutes. The skips cover optional runtime tests in the base environment.
- After the final interpretation fixes: **45 service/API/compute/archive/architecture tests passed**, plus **14 real CPU attention/inference tests** in the isolated training environment. These targeted checks include tests already represented in the full suite; the counts should not be summed as unique tests.
- Frontend: **306 tests passed**; TypeScript and production Vite build passed.
- Ruff passed. A separate import check confirmed no model or imaging libraries load at API import time.
- The final wheel was built and checked against the current clinical, attention, viewer and frontend source files. It contains the imaging extra and the two new module routes. Final cleanup UI checks also passed (8 tests).
- New regressions cover exports above the old 64 MiB limit, atomic output failure, pyramid geometry, indexed mean/member paging, tampering, resume, cancellation, archived links, undefined statistics, viewport bounds, overlapping patches and authenticated image retrieval.
- Browser checks used an offline build of the real components and controlled local fixtures. No application server was started or restarted.

The frontend build retains the existing advisory about a JavaScript chunk larger than 500 kB. This does not block compilation or packaging.

## Deliberate limits

Clinical utility covers fixed binary/multiclass classification outcomes. It does not implement censoring-adjusted survival analysis, causal treatment benefit, monetary cost-effectiveness, bootstrap AUC/Brier intervals or calibration-slope fitting. The report and module guide explain these limits.

Interpretation currently supports native ABMIL and HDF5 patch vectors with level-0 XY coordinates. Features must already exist. Separate coordinates without an embedded counterpart require an explicit row-order attestation. Synchronous review is limited to 45 seconds; per-slide limits are 2 million patches and 8 GiB of feature data. The viewer renders at most 100,000 patches per viewport, with an explicit partial-overlay message and full exports. OpenSlide views are bounded to 2048 pixels; raster fallback is bounded to 32 megapixels. See [the module guide](../../CLINICAL_INSIGHTS.md).

## Screenshots

- [Desktop roadmap](roadmap-desktop.png)
- [Desktop clinical report](clinical-desktop.png)
- [Mobile clinical report](clinical-mobile.png)
- [Real synthetic ABMIL attention overlay](attention-desktop.png)
