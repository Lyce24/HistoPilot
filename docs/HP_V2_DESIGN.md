# HistoPilot v2 — verdict and design

> Historical branch audit and proposal. The independent [main integration review](V2_INTEGRATION_REVIEW.md) supersedes its implementation status and corrects several findings. In particular, main retains fixed-class macro-F1, repairs legacy hash compatibility, preserves executable split machinery, and adds navigation and recovery safeguards. Later phases below remain proposals.

Branch `hp_v2`, started 2026-09-11 from the main working tree (including its uncommitted
work, captured as the baseline commit). This document is tracked at the repository root
because `docs/` is gitignored.

## 1. How this verdict was produced

- Static review of every module in `histopilot/` and `web/src/` (three independent audits:
  backend logic, ML/training path, UI/UX), each claim re-checked against the code before it
  was kept. Findings are marked **confirmed** (path read end-to-end) or **plausible**.
- Checks run on the main working tree: `ruff` clean; `pytest` 1219 passed, 14 skipped,
  **1 failed** (`test_control_service_import_does_not_load_ml_libraries`: the control service
  imported `h5py` through `storage/packed.py`); `tsc --noEmit` clean; **vitest cannot start**
  on this workstation (Node 18.19 installed, `package.json` requires ≥22.12; rolldown needs
  `util.styleText`). UI unit tests are therefore unverified here; install Node 22+ to run them.
- No jobs were launched, no data touched.

## 2. Verdict in one paragraph

The backend is unusually careful: content-addressed frozen records, journaled publication,
patient-grouped splits enforced three times (protocol build, plan validation, worker),
validation-only checkpoint selection, exactly-once OOF assembly, checkpoint hashing, and
dev/test overlap blocking. That rigor is the asset to keep. The product problem is that the
UI exposes every internal object as a user concept: a pathologist must learn ~30 nouns
(dataset version, protocol, cohort, split, feature source, pack, bundle, loading policy,
experiment, batch, configuration, run, execution, predictor, ensemble, refit plan, test
cohort, evaluation plan, bulk evaluation…) and pass ~10 name/review/freeze ceremonies to get
one AUROC, three of which eject them back to the roadmap. Second, several "immutable"
hashes include presentational data, so records go stale when nothing scientific changed.
Third, roughly 12% of the UI and ~2,400 lines of split machinery are demo-only or
unreachable by any executable path.

## 3. Findings

### 3.1 Backend logic (ranked)

| # | Sev | Finding | Evidence | Status in hp_v2 |
|---|---|---|---|---|
| 1 | High | Renaming an experiment wedges every unpublished refit plan: the live draft name is embedded in the evidence manifest whose hash gates launch and publish. | `application/predictors.py::_experiment`, `refits.py::_verify_sources`, `refits.py` `provenanceHash` | **Fixed** (`evidence_hash` strips display fields; legacy hashes accepted) + test |
| 2 | High | Test cohorts go stale when a *warning message* is reworded: `findings`, `canFreeze`, `executionEnabled` were hashed into `previewHash`; `_resolve` then blocks every evaluation. | `application/evaluations.py::_prepare/_resolve` | **Fixed** (`preview_hash` covers scientific keys only; legacy hash accepted on read) + test |
| 3 | High | Interrupted evaluation runs can become permanently unresumable: `gpuIds` derived from *current* CUDA availability, resume rejects a changed plan, fresh launch rejects existing state, and the evaluation ID has no nonce. | `evaluation_runs.py:249-255`, `compute_jobs.py:185,234-239` | Open (design: job nonce + plan resolved at launch, not at resume) |
| 4 | High (sci) | Test-cohort label mapping is never compared with the development mapping (`target` copied from the protocol makes the check tautological); an inverted raw→class map silently inverts AUROC. Empty-string semantics differ between protocol and cohort. | `evaluations.py:116-121,156-157` vs `protocols.py:868-871` | Open (design §6) |
| 5 | Med | 7 of 8 split strategies and 3 of 4 split versions cannot be executed: training/predictors accept only `split.version==4 && mode=="kfold"`. ~2,400 lines of `protocols.py`, `modern_splits.py`, `explicit_pools.py`, `development_splits.py` serve preview only. | `training.py:160-169`, `predictors.py:363-372`, `development.py:454-461` | Open (design §6) |
| 6 | Med | Full-content verification on every read: dataset checksums on `get_dataset`, `quick_check` + DDL comparison on every store call, `stat` of every feature file on every `bundle.get()`, checkpoint re-hashing twice per evaluation. Synchronous in HTTP handlers. | `storage/scientific.py`, `feature_packs.py:748-796`, `bulk_evaluations.py:196` | Open |
| 7 | Med | Archive semantics differ per service: `get_*` only rejects trashed, so single evaluations run against archived predictors while bulk refuses. | `storage/scientific.py::_visible`, `bulk_evaluations.py:746-752` | Open |
| 8 | Med | Tag reservation leak: a `preparing` publication whose draft was trashed reserves its tag forever. | `storage/scientific.py::_recover_locked`, `_check_version_tag` | Open |
| 9 | Med | Lifecycle sidecar is an unbounded, fully replayed log capped at 16 MB with no compaction. | `storage/lifecycle.py:208-220` | Open |
| 10 | Low | Control service imported `h5py` at module import (architecture test failing). | `storage/packed.py` | **Fixed** (lazy import) |
| 11 | Low | `PATCH` on an experiment with notes only sets `inputs: null`. | `model_experiments.py:164-169` | Open |
| 12 | Low | Stale `ARCHITECTURE.md` (says execution unimplemented, jobs return 501). | root | Open |

Naming collisions confirmed in code: "experiment" = project (URL, Start page), draft kind,
model-experiment record, demo spec; "configuration" = any frozen manifest, a training
candidate, and `ProjectConfig`; "batch" = mil-batch, evaluation-batch, predictor build;
"cohort" = demo cohort, evaluation cohort, and (README) the protocol; "test" partition = OOF
assessment fold in split v4 but external test in v3. Six copies of the canonical-JSON hash
helper exist with differing `ensure_ascii`.

### 3.2 Training / inference path (ranked)

| # | Sev | Finding | Evidence | Status |
|---|---|---|---|---|
| 1 | Med | Resume window: if a worker dies after `trainer.fit` returns but before `result.json` is written, resume re-enters `fit` and an early-stopped run trains one more epoch, moving `last.ckpt` and possibly `best.ckpt`. | `training/fold.py:640-665`, `workers/train_batch.py:252-255` | Open (design: `fit-complete.json` marker → prediction-only resume) |
| 2 | Med | `macroF1` averaged F1=0 for classes absent from the fold/cohort while `balancedAccuracy` filtered them. | `training/module.py`, `training/inference.py` | **Fixed** (present classes only) |
| 3 | Med (sci) | Model selection on test is documented, not prevented: all active predictors can be evaluated on a cohort, each cohort carries its own threshold, no pre-registered primary predictor/operating point; OOF accuracy uses argmax while test accuracy uses the cohort threshold. | `bulk_evaluations.py:796-881`, `schemas/evaluations.py:26` | Open (design §6) |
| 4 | Med (sci) | Slide-ID fallback grouping and "independent" patient namespaces are warnings, not opt-ins. | `protocols.py:258-268,664-685`, `evaluations.py:100-109` | Open |
| 5 | Low-Med | Code fingerprint omits `schemas/development.py` (recipe defaults/validators) and histopilot version / git SHA / CUDA / cuDNN. | `workers/training_process.py:780-799` | Open |
| 6 | Med (perf) | Sampled bags use h5py point selection (`source[selection]`) and two `stat` sweeps per item per epoch; training is loader-bound. | `datasets/mil.py:340,374,384` | Open |
| 7 | Low | Pooled OOF metrics only; no per-fold / across-seed dispersion for configuration comparison. | `workers/train_batch.py:194` | Open |

Verified as correct: ABMIL masking and gated attention, float32 softmax under autocast,
hashed deterministic patch/slide-order sampling with epoch carried in the sampler index,
patient-balanced loss weights, validation-only checkpoint selection, assessment set not
opened until fit ends, tie-correct AUROC/AP, `None` rather than 0 for missing-class metrics,
refit trained on the union of development partitions with fixed budget, ensemble member
checkpoint re-hashing, dev/test slide and patient overlap blocking, lease registry with
pid+startTicks+bootId identity, SIGTERM→SIGKILL escalation, archive-on-launch resume.

### 3.3 UI / UX (ranked by impact on intuitiveness)

1. **Five names for two objects.** Protocol / Targets & splits / cohort / "Target & split
   version" / "Development protocol" are one thing; bundle / feature version / feature source
   / verified features are another. `?experiment=` in the URL means *project*.
2. **Ten ceremonies for one model.** Six versioned freezes (dataset, protocol, hidden feature
   source, bundle, batch, test cohort) plus four review-and-confirm gates. Three freezes
   navigate to `#overview` instead of the next step.
3. **Hidden freeze.** "Continue to bundle preparation" silently freezes a feature-source
   configuration with an auto-generated hash tag that then pollutes every picker.
4. **Expert options at the same level as required ones** (packs, mmap, precision, GPU IDs,
   loader workers, epoch percentile) on three pages; a TRIDENT user never needs packs.
5. **Raw enums on screen**: `binary_classification`, `kfold`, `completed · active`.
6. **Dead tools** in every project: Slide explorer (empty state) and Provenance (three counts).
7. **Five stepper/tab implementations, 28 CSS files, 7 design tokens vs 776 hard-coded hex
   colours**, Tailwind imported but ~47 utilities used. Each page feels like a different product.
8. **Demo mode is a second app**: ~2,900 lines of synthetic pages and a parallel roadmap
   state machine; every real user sees "Synthetic demo / Saved locally" pills and disclaimers.
9. **Polling, not events**: 23 `refetchInterval`s; JobTray, Build predictors and the bulk
   runner poll the same lists at 3–15 s; no completion notification.
10. **Dead-end**: evaluation requires the test cohort's protocol *and* development bundle to
    match the predictor exactly; a cohort built first with a different bundle shows "No test
    cohort uses this predictor's development protocol" with no fix path.

Strengths to keep: accessibility (skip link, focus management, `inert` drawer, ARIA roles,
`NumericField` validity), the roadmap's honest "evidence, not form state" semantics,
`FreezeVersionDialog` retaining entries after conflicts, review-hash-then-freeze everywhere.

## 4. v2 principles

1. **One noun per concept, and the UI noun is the code noun.** Table in §5.2.
2. **Save, don't ceremonialize.** Records are still immutable and content-addressed, but
   versions are auto-numbered (v1, v2…) with an optional note; tags become optional aliases.
3. **Next step is always one click away.** Every save lands on the next module, with a
   persistent progress strip; the roadmap becomes a compact header, not a destination.
4. **Hash scientific content only.** Findings, prose, capability flags and display names
   never enter an identity or freshness hash.
5. **Executable-only surface.** Nothing can be frozen that no worker can consume.
6. **Verify on write, trust on read.** Expensive verification happens at publish/launch and
   is cached by (path, stamps); reads are cheap.
7. **Advanced is a drawer, not a section.** Defaults are named in plain language
   ("Read features from your folder"); everything else is behind one Advanced toggle per page.
8. **Pre-registration by construction.** A study declares its primary model and operating
   point from development data before any test metric is shown.

## 5. v2 information architecture

### 5.1 Navigation

```
Projects
└── <project>
    ├── Data            slides & labels table (Datasets today) + Features tab (attach/extract, coverage)
    ├── Study design    target · eligible records · split strategy (Targets & splits today, executable strategies only)
    ├── Training        experiments = design + features + training settings + runs + results (one scrolling page)
    ├── Models          library of ensembles/refits; "Build model" opens from an experiment's results
    ├── Evaluation      left: test sets (define/verify)   right: evaluations (model × test set) with curves and comparison
    └── Tools           Cleanup · System (Slide viewer and Provenance return when they render real content)
```

Progress strip at the top of every page: Data → Design → Features → Train → Models → Evaluate,
each chip showing Not started / Draft / Saved and linking to the module. The DAG roadmap page
is retired; its "next step" logic (`lib/roadmap.ts::buildRoadmap`) is kept and feeds the strip.

### 5.2 Vocabulary

| v2 name | Replaces |
|---|---|
| Project | experiment (URL/Start), workspace |
| Dataset v1, v2… | dataset version, import draft, frozen dataset |
| Features (e.g. "UNI 20× v1") | feature source, feature configuration, feature bundle, feature version; pack becomes an internal storage detail |
| Study design v1 | protocol, targets & splits, cohort / protocol, target & split version, development protocol |
| Experiment | experiment + batch + development plan + MIL plan |
| Variant | configuration / candidate / grid row |
| Run | run, execution, fold run |
| Model (type: Ensemble, Refit) | predictor, frozen predictor, refit plan/job, published predictor |
| Test set | test cohort, evaluation cohort |
| Evaluation | evaluation plan/record, model evaluation, bulk evaluation batch |
| Save (immutable) | freeze, publish, save plan |
| Draft / Saved / Running / Done / Failed | draft, frozen, current, verified, complete, completed, finished, scheduled, planned |
| Version note | commit note; version tag becomes optional |

### 5.3 Target user journey (CSV + TRIDENT folder → evaluation)

| Step | v1 today | v2 target |
|---|---|---|
| Create project | name + storage folder + optional roots | same |
| Data | 3-step wizard, name & freeze dialog, bounce to roadmap | 2 steps (file + ID mapping; preview), **Save** → lands on Features tab |
| Features | attach → inspect → hidden source freeze → packs → bundle name & freeze | attach or extract → coverage → **Save**; packing under Advanced |
| Study design | protocol name + dataset + 4 steps + second name in dialog | target → eligible records → split (k-fold default) → **Save** → lands on Training |
| Training | create experiment → inputs tab → batch form → review → save plan → launch (3 tabs) | one page: pick design + features (pre-filled when unique), settings, **Run**; runs and OOF results appear below |
| Models | separate module, review + acknowledge, refit tab, publish | "Build model" button on results; ensemble default; refit trains and auto-publishes on success |
| Evaluation | Test cohorts module + Evaluate module, exact-match dead-end | one page: define test set (suggests the matching design), pick model(s), **Evaluate**; results with ROC/PR and per-model comparison |

Concept count drops from ~30 to 9; saves from 10 to 5; no navigation to a roadmap page.

## 6. Backend simplification plan

1. **Identity vs freshness** (done for cohorts and refits in this branch): extend the same rule
   to `protocols.py::_preview` (`findings` and role prose are hashed today) and batch previews.
2. **One job state machine**: a single `Job` record {kind, planHash, nonce, session, process
   identity, status, operations} shared by training, compute jobs, packing, extraction and bulk
   evaluation, with one reconciler run at startup and on read. Resolve device/GPU bindings at
   launch time and store them in the job, never recompute them on resume (fixes backend #3).
3. **Executable-only protocols**: keep split v4 k-fold (and an explicit held-out test option
   when a worker consumes it); delete v1–v3 readers, monte_carlo/LODO/nested/imported/rules and
   pool machinery. `protocols.py` should shrink from 1,568 to ~400 lines.
4. **Shared target definition** (field + raw→class map + missing policy) referenced by both
   the study design and each test set; require an explicit "remap" acknowledgement when a test
   set's mapping differs (fixes backend #4); unify empty-string semantics.
5. **Pre-registered evaluation**: an evaluation record names the primary model and the
   operating point (derived from OOF: fixed threshold, Youden, or target sensitivity) before
   the job runs; further models on the same test set are labelled *exploratory* in exports.
6. **Verify on write**: cache dataset/feature verification keyed by (path, dev, inode, size,
   mtime_ns); run `quick_check`/DDL comparison at startup only; hash checkpoints once per
   evaluation; move bulk submission off the request thread.
7. **Archive semantics in one place**: `store.require_usable(ref, allow_archived)`; drop
   per-service `include_inactive` special cases.
8. **Publications and jobs as lifecycle nodes**; release tag reservations when the owning
   draft is trashed; compact the lifecycle log.
9. **One canonical hash helper** (`storage/canonical.py`) and one `{id, contentHash}`
   reference shape; delete the six copies.
10. **Remove dead layers**: `domain/`, `ports/`, `application/{datasets,experiments,audits,jobs}.py`,
    `workers/supervisor.py`, `contracts.experiment`, synthetic `/workspace` routes, 501 stubs;
    rewrite `ARCHITECTURE.md` from the real flow.
11. **Training**: `fit-complete.json` marker → prediction-only resume; hash `schemas/` and
    record histopilot version / git SHA / CUDA / cuDNN; read contiguous bags and index in NumPy
    instead of h5py point selection; per-fold and across-seed mean ± SD next to pooled OOF;
    make slide-ID fallback and independent namespaces explicit opt-ins.

## 7. UI implementation plan

Phase A (this branch): vocabulary consistency, next-step navigation, enum labels, project URL
parameter with the old alias, hide placeholder tools. Phase B: `ui.tsx` primitives
(`Button`, `Field`, `Select`, `Tabs`, `Stepper`, `StatusBadge`, `Dialog`), status colours as
tokens, collapse the five tab/stepper CSS implementations, delete demo pages and
`store/ui.ts`, replace Tailwind-or-hand-CSS ambiguity with one. Phase C: merge Test cohorts
into Evaluation, fold Build predictors into Training results, one-page experiments, progress
strip replacing the roadmap page, auto-versioning with optional tag. Phase D: SSE job events →
one notification centre, delete duplicated polling. Phase E: real slide viewer with attention
overlays from saved fold checkpoints, ROC/PR/calibration plots, exportable evaluation report.

## 8. What this branch changes (Phase A + robustness)

Backend
- `storage/packed.py`: `h5py` imported lazily; the control service no longer loads HDF5 libraries (architecture test passes).
- `training/module.py`, `training/inference.py`: macro-F1 averages present classes only, consistent with balanced accuracy.
- `application/predictors.py`: `evidence_hash` / `evidence_current` — the live experiment name no longer decides whether reviewed evidence is current; pre-v2 hashes still verify.
- `application/refits.py`, `application/predictor_builds.py`: use the stable evidence hash for plan creation, launch, publish and reuse; the refit `provenanceHash` no longer depends on the name.
- `application/evaluations.py`: `previewHash` covers scientific keys only; cohorts frozen before v2 remain current.
- Tests: refit survives an experiment rename; pre-v2 refit hash accepted; reworded findings and legacy cohort hashes stay current.

UI
- `?project=<id>` is the workspace parameter; `?experiment=<id>` links keep working (`projectFromUrl`).
- After freezing a dataset the app opens Targets & splits; after a protocol, Slide features; after a bundle, Experiments.
- `lib/labels.ts`: task, unit and split-mode enums rendered as labels.
- "Cohort / protocol" and "Target & split version" → "Development protocol"; "Feature version" → "Feature bundle".
- Local projects no longer list Slide explorer and Provenance in the sidebar (routes remain).

## 9. Running hp_v2 on this workstation

The branch is a git worktree at `~/projects/HistoPilot-hp_v2` with its own `.venv`
(`uv sync --locked`) and symlinks to main's `web/node_modules` and `.venv-training`.
The system Node is 18; the UI toolchain needs 22+, so run npm commands through a
Node 22 binary fetched by npx (cached after the first run):

```bash
# terminal 1 — control service (add --data-root as needed)
cd ~/projects/HistoPilot-hp_v2
uv run histopilot serve --dev --no-browser --data-root /mnt/d --data-root /mnt/wsl/oceanpath-hot

# terminal 2 — Vite dev server, then open http://127.0.0.1:5173
cd ~/projects/HistoPilot-hp_v2/web
npx -y -p node@22.12.0 npm run dev

# checks
npx -y -p node@22.12.0 npx vitest run      # 275 UI tests pass
npx -y -p node@22.12.0 npm run typecheck
cd .. && uv run pytest -q                  # backend suite
```

Both checkouts share `~/.histopilot/workspace` and `~/.histopilot/config.toml`, so
projects opened from hp_v2 are the same records main sees. Do not run the main and
hp_v2 services on the same port at the same time. For a packaged build:
`npx -y -p node@22.12.0 npm run build` in `web/`, then `uv run python scripts/bundle_web.py`.

## 10. Decisions for the owner

1. Retire the synthetic demo mode in favour of a "seed sample project" action? (Removes ~2,900 UI lines and the dual roadmap logic.)
2. Auto-numbered versions with optional tags, or keep required tags?
3. Keep any non-k-fold split strategy for v2, or delete until a worker consumes it?
4. Adopt pre-registered primary model + operating point as a hard gate before test metrics are shown?
5. Node 22+ on this workstation so `vitest` can run in CI and locally.
