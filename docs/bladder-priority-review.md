# Bladder priority review and next implementation plan

The next milestone should be **create Bladder → import the actual manifest and slide inventory → review mapping → freeze → reopen**. Then attach the existing UNI features, resolve patient identity, freeze an explicit evaluation protocol, and enforce preflight before one real baseline run. The start/load page is already implemented; another navigation redesign would not unlock this scientific workflow.

Bladder is the reference for all upcoming product designs and acceptance flows. Keep the core contracts generic, but defer CRC-specific task hierarchies, master-split inheritance, and model expansion. The older [two-P0 proposal](p0-project-import-target-split-design.md) remains background; this review supersedes its implementation order, central scientific-storage proposal, patient-only prediction assumption, and CRC delivery requirements.

The follow-up [data-stage schema design](data-stage-schema-design.md) specifies canonical identifiers, attribute definitions and target roles, combined versus separate CSV/XLSX input, and cBioPortal-inspired exploration with explicit patient/slide counting units.

## 1. What the local evidence establishes

This review inspected the local files on 2026-09-09. It combines complete small-table and filename inventories, all 138 primary UNI HDF5 headers, four WSI metadata probes, sampled headers from alternative feature stores, existing split artifacts/configuration, and the current HistoPilot implementation. A tiny alternative-store value comparison also read nine feature rows by sixteen dimensions; no complete feature arrays or WSI pixels were read. The accompanying [aggregate evidence](evidence/bladder-audit-2026-09-09.json) contains no row-level slide, case, or patient identifiers. No clinical files were changed, no full WSI or feature hashes were computed, and no training or extraction was launched.[^1]

### Slides and metadata

| Input | Observed state | Design consequence |
| --- | --- | --- |
| `/path/to/research-data/slides/blca` | 138 flat `.tiff` files; 138 distinct filename stems; no zero-byte files; 242,914,452,312 bytes, approximately 242.9 GB / 226.2 GiB | Reference files in place. Inventory completion must be separate from expensive content verification. |
| `/path/to/research-data/manifests/BLCA/BLCA.xlsx`, `Sheet1` | 144 rows; `Case`, `De ID`, `Location`, both WHO columns, comments and demographics | Support XLSX sheet selection and preserve original columns. Retain case information without calling it patient identity. |
| `/path/to/research-data/manifests/BLCA/blca.csv` | 138 rows; `De ID`, `WHO 2022`, `WHO 1973`, `Binary WHO 2022`, `k_fold`; exact match to all 138 TIFF stems | This is the smallest useful first import. Preserve identifiers as strings and map them explicitly to filenames. |
| `blca_k_fold.csv` and `blca_k_fold.xlsx` in the same manifest directory | CSV is byte-identical to `blca.csv`; the derived XLSX has the same table content | These are alternative representations of one prepared table, not three cohorts to concatenate. |
| Raw-to-prepared reconciliation | Six raw rows have no corresponding current TIFF or prepared row; two per WHO1973 grade, three low and three high WHO2022 labels | Show an exclusion/unmatched ledger. Do not silently discard the six records or suggest their absence is a label error. |

All 138 prepared IDs match raw workbook IDs exactly; no identifier normalization is needed for these inputs. All currently matched WHO labels agree between the raw and prepared tables. The matched cohort has this distribution:[^2]

| WHO1973 grade | WHO2022 low / binary 0 | WHO2022 high / binary 1 | Slides |
| --- | ---: | ---: | ---: |
| 1 | 33 | 0 | 33 |
| 2 | 54 | 22 | 76 |
| 3 | 0 | 29 | 29 |
| **Total** | **87** | **51** | **138** |

The local notebook documents a historical availability filter and uses `0 if low else 1`. Today's matched values are clean, but this rule would incorrectly classify a future unknown or missing value as high. HistoPilot must use an explicit accepted vocabulary and preserve unmapped values as findings. Its first-dot identifier normalization must likewise not become a generic filename rule.[^2]

Four sampled TIFFs opened for metadata as Philips pyramidal BigTIFF, with 512 × 512 JPEG tiles, 0.25 µm/pixel on both axes, and 40× objective metadata. These are four successful metadata probes, not a full 138-slide pixel-readability audit. A modest metadata/thumbnail adapter is feasible; a large viewer rewrite is not a prerequisite for import. OpenSlide documents Philips detection and its physical-resolution properties.[^1][^3]

### Identity is the main scientific dependency

The user confirmed that **`De ID` identifies a slide/case**. It must not be used as evidence of a unique patient. The raw workbook has 144 distinct `De ID` values but 143 distinct `Case` values. Only one member of the repeated-case pair is present in the current 138-slide cohort. Consequently, the matched table does not expose a repeated-case split overlap, but neither its uniqueness nor the absence of observed overlap proves patient separation.[^2]

No verified patient crosswalk was found in the inspected sources. The number of patients is therefore **unknown**, not 138. This is a blocker for the requested patient-grouped execution guarantee. It does not block storing sources, previewing labels, inspecting features, or freezing a dataset whose identity status is explicitly unresolved.

Separate three concepts in the next design:

| Concept | Bladder interpretation | Proposed treatment |
| --- | --- | --- |
| Input unit | One feature bag per TIFF/slide | Link it to a stable slide ID. |
| Label and prediction unit | Labels currently attached to slide/case records | Start by representing them at that source unit. Make the selected prediction/evaluation unit explicit. |
| Split grouping unit | Patient | Require a verified slide/case-to-patient crosswalk before patient-grouped split readiness. |

This corrects the earlier proposal's assumption that every P0 target should immediately resolve to one patient label. For a slide-level target, different correctly annotated cases from one patient need not be a label error; they still stay in the same partition. For a patient-level target, conflicting source labels require a declared clinical resolution/exclusion policy. Neither majority voting nor copying the slide ID into `patient_id` is an acceptable implicit resolution. Patient-level metrics also require an explicit aggregation rule; renaming slide-level results is insufficient.

### Existing features make new extraction a lower priority

Use this as the first candidate store:

```text
/path/to/research-data/features/blca/20x_256px_0px_overlap/features_uni_v1
```

All 138 HDF5 files match the current slide inventory. Header inspection of every file found nonempty `features[N, 1024]` float32 arrays and `coords[N, 2]` int64 arrays, with matching row counts. Dataset attributes consistently declare `uni_v1`, 20× and 256-pixel patches. The store occupies 5,010,997,392 bytes, about 5.01 GB, and contains 1,191,038 feature rows in total. Bags range from 1,148 to 39,525 rows; median size is 7,040.5.[^4]

That is strong evidence for prioritizing **existing-feature attachment and validation**. It is not yet a certificate of finite values, correct coordinates, unchanged bytes, or authentic extraction provenance. The current sidecar declares an encoder fingerprint and code revision; those declarations do not independently establish the exact weight bytes that generated the features.

Other stores include UNI v2, Virchow2, ResNet50, and artifact/penmark-processing variants. Several contain 139 files, including one unmatched file relative to this cohort. Equal encoder names and matching IDs do not make different preprocessing runs interchangeable. Keep each candidate separate, show orphans, and never recursively merge them into one feature set.[^4]

The configured `/path/to/research-data/mmap/blca/uni_v1` contains an index archive without the corresponding feature chunks or metadata. Legacy backup packs exist elsewhere, but their `features.npy` files lack NumPy headers and require their own raw-array metadata. Start with the directly inspectable HDF5 store; a saved mmap path alone is not evidence of usable training input.

### Existing splits represent different protocols

| Artifact/protocol | Observed allocation | Readiness concern |
| --- | --- | --- |
| Prepared CSV `k_fold` | 76 grade-2 slides have `-1`; remaining 62 have validation folds of 13, 13, 12, 12, 12 | Notebook uses ungrouped stratified folds. Import as legacy assignments with declared semantics, then audit. |
| `blca_g2test_kfold5_seed42` | Same grade-2 test population size; 62 development slides, five folds | Stored `group_column` is null and every group equals a slide ID. No patient guarantee. |
| `blca_g2test_heldout_seed42` | 49 train, 13 validation, 76 grade-2 test | A single holdout protocol, not the same experiment as five-fold development. |
| `kfold5_seed42` | 14 rows have `fold=-1`; 124 are in folds of 25, 25, 25, 25, 24 | Stored configuration declares no test ratio/filter; these memberships conflict with that declared protocol. |

All three stored split tables match the prepared manifest's current slide IDs and binary labels. The two grade-2 protocols match their declared grade rule. The other 14-slide holdout must remain a protocol-mismatch finding: the inspected current generator/tests explicitly preserve no-test configuration, but the precise cause of this old artifact discrepancy is not established. None of the three stored summaries contains the current generator's split-identity metadata.[^5]

The grade-2 protocol tests 55.1% of the current slide cohort: 54 low and 22 high. Development has 33 low and 29 high. This evaluates performance on grade-2 cases after development on grades 1 and 3; it must not be presented as an ordinary random holdout from the entire cohort. Make it a named selectable protocol. It is a valuable historical reproduction target, but this review does not choose it as the scientific default for every future experiment.

Once patient mappings exist, whole-patient holdout expansion may change these counts. Importing old assignments must either pass the new patient audit or remain an immutable legacy record; repairs produce a new split version. Do not silently move cases and retain the old split identity.

## 2. What is implemented versus what still needs building

| Capability | Current status | Implication |
| --- | --- | --- |
| Start / create / load | Implemented, including chosen storage folder, optional paths/configuration, and explicit demo isolation | Preserve the flow and existing folder-reopen behavior. |
| Source registration | Directory references only | Add typed metadata-file selection and readers. A saved folder is not an imported dataset. |
| Real Dataset page | Saved paths and empty scientific state | Replace the empty state with the actual import/review workflow. |
| Dataset ingestion / audit services | Stubs | The two scientific P0s remain implementation work. |
| Experiment configuration | Initial preferences plus a legacy binary draft schema | Introduce frozen target/cohort/split references and explicit class semantics. |
| Execution | Job submission returns 501; executor is unimplemented | This refusal does not prove that leakage or invalid labels are detected. |

These findings come from the real-mode workspace, filesystem service, application stubs, experiment contract, and API.[^6] Current setup persistence lives in the selected folder's `histopilot-project.json`; the central SQLite database only indexes recent local projects. The original P0 document's central-database authority would break that contract unless accompanied by a restoration mechanism.[^7]

## 3. Ranked next work

The ranking follows dependency order and the amount of real bladder workflow it unlocks. Patient-map acquisition should proceed alongside software work; it is not a reason to postpone the importer.

| Priority | Deliverable | Why now | Acceptance gate |
| --- | --- | --- | --- |
| **P0.0** | Folder-owned scientific persistence, revisions and publication recovery | Import drafts and frozen records must survive opening the experiment from a fresh service registry | Existing setup reopens unchanged; stale edits fail; competing project writers cannot overwrite scientific state; interrupted publication exposes no partial version. |
| **P0.1 — next visible feature** | Real Dataset import: CSV/XLSX + full slide inventory + mapping/review + freeze | Existing files already support the first end-to-end outcome | Create Bladder → preview the 138 matches and any raw-table exclusions → freeze → restart/reopen with identical records and dataset identity. |
| **P0.2** | Identity-aware targets, cohort and explicit split import/generation | `De ID` is not patient identity, and legacy split protocols differ | Missing patient mapping prevents patient-grouped readiness; explicit WHO labels and whole-patient assignments are reproducible and independently audited. |
| **P0.3** | Existing UNI attachment, complete validation and shared preflight | Avoid re-extracting available features; establish the execution barrier | Selected bags are compatible and verified to the declared level; invalid labels, leakage, stale inputs or incompatible features yield zero executor calls through API and CLI. |
| **P1.0** | One durable local baseline run and real evaluation | First scientific value after trustworthy inputs | One pinned MeanPool/classifier baseline produces recoverable job state, held-out predictions and linked artifacts; then add Gated ABMIL. |
| **P1.1** | Focused slide/patch QC, then one extraction adapter | Useful for provenance review and missing/rejected feature stores | Correct slide thumbnails and coordinate transforms; one pinned extraction path whose outputs pass the same feature validator. |
| **P2** | Model matrix, rich explorer, broader study designs and deployment integrations | These do not resolve the present data/identity bottleneck | Extend the proven contracts and execution path. |

### P0.0: preserve folder portability before adding scientific writes

Recommended layout: the selected experiment folder owns a versioned descriptor, a project-local transactional scientific store, and immutable artifact directories. The central database continues to index recent projects and own the explicit demo. Define one authority for each field: keep setup in the descriptor; store scientific drafts, revisions and publication state in the project store; do not independently edit duplicate copies of those fields.

```text
<chosen experiment folder>/
  histopilot-project.json
  histopilot-state.sqlite
  imports/<import-id>/
  datasets/<dataset-id>/
    manifest.json
    sources/                 # snapshots of imported metadata
    tables/                  # canonical records + unmatched/excluded ledger
  targets/  cohorts/  splits/  features/  preflights/  runs/
```

Introduce version compatibility/migration with the first new stored objects, preserving existing settings and references. Add an experiment-folder writer lock plus optimistic revisions. The present service lock protects the central workspace, while the project lock is only in-process; two services using different registries can otherwise mutate the same chosen folder.[^7]

Validate the supported storage filesystem for the proposed database/locking scheme. SQLite WAL requires same-host shared-memory support and does not work over a network filesystem. This is a storage acceptance requirement, not a reason to copy the 243 GB slide collection.[^8]

Use staging plus atomic publication and an explicit recovery journal/state. Commit the visible dataset only when all required files are durable. A retry must produce one version. Reopening must use frozen metadata even if an external source is temporarily unavailable. These are part of P0 freeze/reload, not end-of-roadmap cleanup.

### P0.1: replace the real Dataset empty state

Keep the existing Overview / Dataset / Features / experiment navigation. Within Dataset, use **Sources → Mapping → Review → Freeze**; preserve a server-side draft through browser refresh.

| Step | Bladder content |
| --- | --- |
| Sources | Select `blca.csv` or an XLSX sheet; reference `slides/blca`; optionally reference an existing feature root. Add the raw workbook as a second reviewed source when reconciling case metadata and exclusions. |
| Mapping | `De ID` → slide/case source ID; exact filename-stem matching for the prepared CSV; WHO columns → retained attributes; patient mapping → unresolved or an explicit crosswalk. |
| Review | 138 matched slides, 138 labeled records, 87 low / 51 high; six unmatched raw records when the workbook is included; patient count unresolved. Show the actual join keys, normalization and cardinality. |
| Freeze | Publish immutable source-table snapshots, canonical selected records, mapping decisions, complete inventory and exclusions. Present verification scope separately from frozen status. |
| Reopen | Load the same version and findings; source availability checks do not replace frozen values. |

Support server files before broad upload features. CSV and XLSX matter immediately because those are the actual inputs. Metadata uploads can use the same parser/staging path within P0; Parquet/JSON adapters can follow without blocking this first vertical slice. Large WSI/feature folder upload remains out of scope.

The directory browser is nonrecursive and capped at 200 encountered entries. Do not reuse its response as a dataset inventory. Build a complete scanner with explicit scope, extension filters, per-file path validation and persisted completion status. Tiny metadata previews can be synchronous; scans, content hashing and full feature checks need cancellable, restart-aware CPU operations. That minimal operation lifecycle belongs here, separately from GPU scheduling.[^6]

Hash small source tables and canonical scientific records at freeze. Record external-file identities and the exact level of verification performed. Full WSI hashing should not block a metadata-only import, and unavailable pixels need not block a feature-only training operation. Neither a partial scan nor size/mtime alone may be presented as verified file content. Check the artifacts consumed by each operation, report coverage of duplicate/content checks, and never certify uninspected pixels as duplicate-free.

### P0.2: make target and split semantics explicit

Provide a Bladder preset for **WHO2022 low/high**, preserving `0 → low`, `1 → high`, with high as the proposed positive class. Keep WHO1973 available as a separate categorical attribute and optional three-class target. Presets fill an editor; they do not overwrite the raw values or select the final scientific question.

The patient crosswalk should map each selected slide/case to one stable patient identifier with source provenance. It must distinguish a verified identity mapping from a user-entered column name. Unknown or ambiguous links block patient-grouped execution. Do not infer patients from demographics, file names or the OceanPath configuration's `patient_id_column: De ID`.[^9]

Offer two distinct split workflows: **import and audit an existing protocol**, or **generate a new patient-grouped protocol**. Before freezing, show train/validation/test roles, patient and slide counts, class support, any grade-2 holdout predicate, and whole-patient expansion. Interpret the prepared CSV's `-1` only under the reviewed legacy rule; a numeric sentinel has no universal test meaning.

For new assignments, canonicalize identifiers/order, pin the algorithm and library version, distinguish split seed from training seed, and persist exact memberships plus a semantic assignment hash. Test input-row and directory-order invariance. StratifiedGroupKFold is a candidate for grouped stratification, but its documentation explicitly describes approximate class balance subject to group constraints; a fixed seed is not sufficient evidence of a valid study split.[^10]

Choose whether balancing is measured over patients or slide/case labels; record that choice. For a consistent patient-level target, splitting one canonical row per patient avoids allowing patients with many slides to dominate patient balance. For a slide-level target with heterogeneous case labels, validate grouping and report both patient and slide distributions. In either case, infeasible class support blocks the proposed configuration rather than silently changing the protocol.

Keep final test labels out of threshold, epoch and model selection. The proposed grade-2 protocol uses development folds for selection and reserves its fixed test population for the declared final evaluation. Historical evidence counts are slide counts; patient support and resulting group-expanded holdout counts remain unknown until identity is resolved.

### P0.3: validate existing features and enforce one execution barrier

The first attachment adapter should read HDF5 dataset attributes as well as root attributes; this actual primary store has no root attributes. Allow an explicit feature/coordinate dataset mapping and recognize separate coordinate filenames such as `<slide>_patches.h5`. Preserve dotted identifiers; do not guess joins by stripping everything after the first dot.

Separate three statuses: **discovered**, **structurally inspected**, and **content/provenance verified**. Full validation must cover every selected file, check compatible dimensions/dtypes, nonempty and finite features, feature-to-coordinate alignment, duplicate mappings, content identity and source drift. Read arrays in bounded slices; h5py exposes shape/dtype independently of payload and supports sliced/chunked reads.[^11]

Record encoder/checkpoint declarations, extraction settings, preprocessing variant, coordinate units and validation evidence. Unknown extraction details remain unknown. Before a training release, decide the explicit policy for externally supplied features with verified bytes but incomplete extraction provenance; they must not receive a complete extraction-reproducibility claim. Recover provenance where possible, and never silently replace missing evidence with a folder name.

The shared preparation boundary must resolve dataset, target, cohort, split, feature set and specification revisions together. It runs immediately before submission through API and CLI. A historical green report becomes stale after any relevant input/specification change. Findings need a stable code, affected count, reason and repair action, for example `PATIENT_ID_UNRESOLVED`, `SPLIT_PROTOCOL_MISMATCH`, `LABEL_UNMAPPED`, or `FEATURE_CONTENT_UNVERIFIED`.

Expose `scientificReady` separately from `executionReady`. Test the scientific barrier with a recording fake executor: leaking or invalid inputs cause **zero runnable jobs and zero executor invocations**. Today's unconditional HTTP 501 is not that test.[^6]

## 4. First reviewable implementation sequence

1. **Storage decision and typed source preview.** Add versioned metadata-file references and CSV/XLSX column/sheet preview in the real Dataset page. Preserve directory references and old experiment folders. Implement the ownership/revision boundary needed for these new writes. A sample preview clearly reports that complete inventory/freeze has not happened.
2. **Complete import and freeze/reopen.** Persist full inventory, reviewed joins, exclusions and parser/source fingerprints; publish immutable versions with retry/recovery behavior. This is the first visible P0-A release.
3. **Two parallel branches after dataset contracts stabilize.** Implement target/identity/split workflows and the UNI feature adapter independently against frozen dataset IDs. Collect the patient crosswalk while this work proceeds.
4. **Shared preflight and specification update.** Join those branches, use the same API/CLI semantics, and demonstrate blockers with a fake executor. This is P0-B; real Bladder patient readiness remains conditional on actual identity evidence.
5. **One baseline and durable execution.** Run a small smoke fixture, then one explicitly selected bladder protocol with verified inputs. Persist lifecycle, logs, checkpoints, predictions, metrics and code/environment identity. Add ABMIL only after the simpler baseline path is inspectable.

The first importer acceptance screen should say something like **“138 slides · 138 labels · patient mapping needed”**. Add **“UNI headers checked 138/138”** only after HistoPilot's feature adapter itself performs and persists that inspection; this external audit is not application validation. “138 patients” or “ready to train” would be incorrect based on today's evidence.

## 5. Acceptance tests that matter

Use small synthetic bladder-shaped fixtures for automated tests, including multiple slides and cases per patient. The actual matched cohort, with one observed case per available slide, cannot establish grouping correctness by itself. Real-data acceptance can be a deliberate read-only smoke review after the importer ships; it should not make CI depend on `/mnt/d`.

| Test | Required evidence |
| --- | --- |
| Folder lifecycle | Existing v1 setup opens; a scientific project reopens from a fresh central registry or after a permitted move; missing source mounts do not erase metadata. |
| Concurrent edit / recovery | Stale revisions fail; another project writer cannot overwrite a draft; interruption during freeze never yields a half-visible version. |
| Parser and mapping | CSV/XLSX preserve string IDs and column names; known suffix rules are explicit; duplicate/ambiguous joins and outside-root references produce findings. |
| Complete inventory | More than 200 files and nested scopes are scanned completely; canceled/truncated scans cannot freeze. |
| Cohort accounting | Matched/unmatched/excluded totals reconcile; adding a previously missing slide creates a child version and triggers new identity/split checks. |
| Labels and units | Unknown WHO values never become high; patient-target conflicts block unless explicitly resolved; valid heterogeneous slide labels remain grouped by patient. |
| Reproducibility | Identical inputs/settings under shuffled row/directory order produce the same assignment hash; reload reads stored assignments. |
| Leakage | Multiple slides/cases per patient stay together; imported cross-partition patients and duplicate artifact aliases block the relevant operation. |
| Protocol mismatch | A no-test configuration paired with fixed-test rows is rejected; grade-2 whole-patient expansion and class support are reported. |
| Features | Missing/empty/NaN/wrong-dimension bags, row mismatches, unexpected orphans and drift cannot silently change eligible membership. |
| No bypass | API and CLI reject unresolved identity, invalid labels and leaking splits before executor invocation; demo IDs cannot enter a real executable specification. |

## 6. What to defer, and what remains uncertain

Defer new encoder families, a full patch/attention explorer, distributed scheduling, arbitrary joins/transformation scripts, survival/regression tasks, and CRC-specific task derivation. A small real thumbnail/metadata view can support import QC without becoming a separate viewer project. Keep CRC as optional existing-demo/contract regression coverage; use bladder fixtures for upcoming product acceptance.

Three decisions remain data/protocol dependencies rather than UI polish: the authoritative patient crosswalk; the intended first evaluation protocol (grade-2 reproduction or a newly specified grouped study); and acceptable provenance requirements for imported legacy features. None is needed to start the importer. The first is required before claiming the requested patient grouping guarantee; the latter two must be explicit before a real run.

The observed labels are clean, but the full clinical identity model, all feature values, exact extraction weights, WSI pixel integrity, and duplicate image content remain unverified. The report's recommendations are a delivery judgment based on current local evidence, not a claim that this dataset is already ready for scientific execution.

## Sources and evidence

[^1]: [Aggregate inspection snapshot](evidence/bladder-audit-2026-09-09.json), built from read-only inventories and metadata probes. Four sampled WSIs; all 138 primary UNI headers; alternative stores sampled. Observation date: 2026-09-09.
[^2]: Local primary inputs: [raw workbook](/path/to/research-data/manifests/BLCA/BLCA.xlsx), [prepared CSV](/path/to/research-data/manifests/BLCA/blca.csv), [derived CSV](/path/to/research-data/manifests/BLCA/blca_k_fold.csv), [derived workbook](/path/to/research-data/manifests/BLCA/blca_k_fold.xlsx), and [preprocessing notebook](/path/to/research-data/manifests/BLCA/blca_preprocess.ipynb). Patient-ID semantics were clarified by the user during this review: `De ID` means slide/case.
[^3]: [OpenSlide Philips format documentation](https://openslide.org/formats/philips/). Used to interpret the sampled vendor/physical-resolution metadata, not to certify all files.
[^4]: [Primary UNI store](/path/to/research-data/features/blca/20x_256px_0px_overlap/features_uni_v1), [current feature root](/path/to/research-data/features/blca), [backup BLCA feature root](/path/to/research-data/backup_features/BLCA_AI), and [configured mmap root](/path/to/research-data/mmap/blca/uni_v1); aggregate coverage/header findings retained in the snapshot.
[^5]: [Stored bladder splits](/path/to/research-data/splits/blca), particularly each `summary.json` and assignment table plus `kfold5_seed42/_run/config.yaml`; [OceanPath split core](../../OceanPath-main/src/oceanpath/splitting/core.py), [workflow](../../OceanPath-main/src/oceanpath/workflows/splitting.py), [kfold configuration](../../OceanPath-main/configs/splits/kfold5.yaml), and [split tests](../../OceanPath-main/tests/test_splitting.py). This compares existing artifacts to inspected current code; it does not establish which historical code produced them.
[^6]: [Real workspace service](../histopilot/application/project_workspace.py), [Dataset page](../web/src/pages/LocalWorkspace.tsx), [filesystem service](../histopilot/storage/filesystem.py), [dataset stub](../histopilot/application/datasets.py), [audit stub](../histopilot/application/audits.py), [v1 experiment contract](../histopilot/contracts/experiment.py), and [API](../histopilot/api/app.py).
[^7]: [Folder ownership implementation](../histopilot/application/project_workspace.py), [workspace contract](workspace.md), [project lifecycle tests](../tests/test_projects.py), [service lock](../histopilot/service_lock.py), and [current database initializer](../histopilot/storage/database.py).
[^8]: [SQLite write-ahead logging documentation](https://sqlite.org/wal.html), especially same-host/shared-memory requirements and journal-mode capability checks.
[^9]: [OceanPath bladder data mapping](../../OceanPath-main/configs/data/blca.yaml), [custom five-fold configuration](../../OceanPath-main/configs/splits/blca_custom.yaml), and [custom holdout configuration](../../OceanPath-main/configs/splits/blca_heldout.yaml). These record software conventions, not independently verified patient identity.
[^10]: [scikit-learn StratifiedGroupKFold documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html). HistoPilot must still define ordering, units, feasibility and persisted assignment/version semantics.
[^11]: [h5py dataset documentation](https://docs.h5py.org/en/stable/high/dataset.html), covering shape/dtype metadata and bounded slicing/chunk iteration.
