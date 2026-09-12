# Implementation handoff

The start/create/load flow is implemented: users choose an experiment folder, optionally save source paths and initial settings, and enter the existing workspace. P0.1 import/mapping/freeze and P0.2 target/grouped-split APIs and UIs are implemented, with existing-feature header attachment. Full tensor/provenance validation and compute remain unfinished. See [implementation notes](p0-import-protocol-implementation.md).

**Bladder is the reference for upcoming designs.** Follow the evidence-backed [Bladder priority review](bladder-priority-review.md) and its [aggregate audit](evidence/bladder-audit-2026-09-09.json). The verified visible path is **create Bladder → import → preview mapping → freeze → reopen**. The older [two-P0 design](p0-project-import-target-split-design.md) supplies background contracts; the review supersedes its storage ownership, prediction-unit assumption, CRC delivery requirements, and implementation sequence.

The observed inputs are 138 matching TIFFs and prepared metadata rows, plus a primary UNI feature store with exact 138-slide coverage and readable headers. The user confirmed `De ID` means slide/case; patient identity is unresolved. Do not report 138 patients or certify the existing splits as patient-grouped.

Keep the five rules in [ARCHITECTURE.md](ARCHITECTURE.md). Build one complete scientific path before expanding encoders or models.

## P0.0 — Scientific persistence in the chosen experiment folder — implemented

`histopilot-project.json` keeps setup ownership; `histopilot-state.sqlite` now stores scientific drafts, revisions, snapshots and publication operations in the chosen folder. A cross-process project writer lock coordinates distinct service registries. Scientific draft updates require an expected revision; publication uses staged artifacts, content hashes and interruption recovery. Existing descriptor-only projects initialize scientific storage without changing their setup. Unknown/incompatible stores are rejected rather than reset.

Draft write/read and dataset metadata read APIs are available; snapshot publication is an internal application boundary, not a browser dataset-upload endpoint. The importer/freeze UI now uses this publication boundary. See the [implementation handoff](p0-storage-implementation.md), [workspace layout](workspace.md), and [API](api.md). Durability support currently requires POSIX locking/directory synchronization and a compatible local SQLite WAL filesystem.

## P0.1 — Real Bladder Dataset import and freeze — implemented for bounded metadata

Use the [data-stage schema design](data-stage-schema-design.md): canonical `Slide_ID`/`Patient_ID`, reviewed mappings from existing headers, typed entity-owned attributes, combined or separate CSV/XLSX tables, and a persisted dictionary driving basic exploration. Patient identity can remain unresolved during import, but must be verified before patient-grouped execution.

Add typed metadata-file references, CSV/XLSX sheet/column preview, a complete slide scanner, explicit identifier mapping and a reconciled review. Prepared `blca.csv` matches all 138 TIFF stems. Including raw `BLCA.xlsx` requires accounting for its 144 rows and six unmatched records. Preserve source values and string identifiers; use an explicit crosswalk for any joins. Source folders remain read-only references.

Persist import drafts and complete inventory status. Keep the 200-entry directory picker separate from scanning. The current implementation uses bounded CPU scans and rejects over-limit inventories without publication. Restart-aware workers for large scans/full validation/hashing remain an explicit follow-up before GPU scheduling. Freeze canonical tables, metadata snapshots, source/mapping fingerprints and exclusions. Record external artifact verification scope separately from dataset immutability.

**Done when:** create Bladder → preview mapping → freeze → restart/reopen retains identical records and dataset identity; repeat import is deterministic; incomplete scans and stale source/mapping revisions cannot freeze. Unresolved patient identity remains visible without preventing metadata import.

## P0.2 — Targets, identity, cohort and patient splits — implemented

Separate slide/case annotation, prediction/evaluation unit, and patient split grouping. Provide an explicit WHO2022 low/high mapping, with WHO1973 retained separately and available as an optional categorical target. Missing/unmapped eligible labels block readiness; patient-target label conflicts require an explicit policy.

Require a verified slide/case-to-patient mapping before patient-grouped execution. Import old split files as candidates with declared semantics, or generate a new grouped protocol. Distinguish the grade-2 holdout (currently 76 test / 62 development slides) from ordinary random protocols. The stored `kfold5_seed42` contains 14 fixed-test rows despite a no-test configuration; it must not be silently reused.

Persist exact assignments, protocol/algorithm/library versions, seed and content identity. Test row-order invariance, whole-patient holdout expansion and class feasibility. Keep split seeds separate from training seeds. Do not infer patient grouping from unique slide IDs.

**Done when:** the same frozen inputs/settings give identical assignments; multi-slide patients never cross partitions within a fold; unresolved identity, conflicting protocol declarations and leaking imported assignments produce actionable blockers. Actual Bladder patient readiness requires the missing identity evidence.

## P0.3 — Full feature validation and mandatory execution preflight

Attach `/mnt/d/YC.Liu/features/blca/20x_256px_0px_overlap/features_uni_v1` first. This review checked all 138 headers, but the application must perform and persist its own validation. Inspect dataset-level attributes, features/coordinates, coverage, content identity and provenance. Read finite-value checks in bounded slices. Keep preprocessing variants separate and report unmatched files. The configured mmap index does not constitute a complete feature cache.

Join dataset, target, cohort, split, feature set and experiment specification at one preparation boundary shared by API and CLI. Scope checks to the intended operation, distinguish scientific from execution readiness, and invalidate stale reports. Unknown provenance cannot be presented as complete extraction reproducibility.

**Done when:** validated selected features resolve to immutable inputs; invalid labels, unresolved patient grouping, leakage, incompatible/missing features and stale input state block submission; a recording fake executor observes zero calls on rejected requests. HTTP 501 alone is not evidence of scientific preflight.

## P1.0 — One durable baseline and real evaluation

Implement isolated local worker execution with stored lifecycle, progress, logs, process identity, cancellation and restart reconciliation. Start with one MeanPool/classifier baseline on verified existing features; add Gated ABMIL after that path works. Keep CUDA dependencies outside FastAPI. Prepare any training cache explicitly and validate its outputs.

Define model/epoch/threshold selection on development data, final test use, prediction unit, aggregation and uncertainty before comparisons. Persist checkpoints, predictions, metrics and environment/code/input lineage.

**Done when:** one pinned GUI/CLI intent produces inspected held-out results, and run/failure/cancel/interruption states are accurate. A browser disconnect does not cancel the job; partial artifacts never appear complete.

For long-running workstation jobs, check tmux sessions, use a descriptive session with persistent logs, and verify the process. Respect the user's ownership of their terminal server. Checkpoint/resume is needed for reboot resilience; tmux only protects against session disconnection.

## P1.1 — Focused QC and one extraction path

Add real slide metadata/thumbnails and selected patch QC where they help review mappings or feature provenance. Four sampled TIFFs opened as Philips pyramidal BigTIFF in the local reader environment; application support and full pixel integrity are not yet verified. Preserve level-0 geometry and test coordinate transforms before attention overlays.

Then implement one version-pinned TRIDENT PFM adapter for missing or rejected feature stores, using the same validator and provenance contract as imported features. Keep the [OceanPath adapter plan](oceanpath.md) as implementation background.

**Done when:** a small real extraction produces correctly aligned, validated artifacts; failed outputs are unavailable; viewed regions resolve to their slide, coordinates and feature/run identity.

## P2 — Expansion and deployment

Expand model/encoder choices, rich WSI exploration, broader import formats, arbitrary study transformations, CRC-specific shared master/task splits, and distributed backends only after the Bladder path works. Keep the existing synthetic demo isolated and non-executable.

Add broader backup/retention, resource diagnostics and installation capabilities as execution grows. For exposed lab/server deployment, authentication/authorization and HTTPS precede non-loopback binding. Continue bundling the frontend inside the wheel.

**Done when:** extensions preserve frozen input contracts, actual results and reliable packaging; they do not weaken preflight or require disease-specific core logic.
