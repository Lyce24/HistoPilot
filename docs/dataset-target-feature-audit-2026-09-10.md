# Dataset, target, and feature workflow audit — 2026-09-10

The audited workflow now blocks several concrete routes to incorrect patient grouping, target leakage, stale publication, and misleading extraction or pack verification. Dataset and protocol recovery also preserves deliberate user choices. These checks improve the evidence behind a frozen configuration; they do not establish biological independence or eliminate every possible source of leakage.

This report covers the current workspace implementation, regression tests, and an isolated browser exercise. No real research data were modified. Feature-loading performance is documented separately in [the matched loading benchmark](feature-loading-benchmark-2026-09-10.md).

## Dataset import and identity

New fixes address the following reproduced defects:

- **Empty identities accepted through missing-value settings.** An empty slide ID or patient join key cannot become a valid identifier when the user removes the empty-string missing token. Whitespace-only IDs are rejected. Patient IDs such as `P1` and ` P1 ` cannot silently become separate patient groups: their collision blocks freezing. A consistently padded ID produces a warning while preserving the original value; the importer does not silently normalize identity evidence.
- **Missing patient-source matches hidden by a successful import.** Unmatched retained slides now produce an explicit warning. Missing patient attributes remain missing; the importer does not invent a patient match. Existing patient identity can remain valid when only supplemental attributes are absent.
- **Excel error cells treated as real labels or IDs.** Native XLSX error cells in mapped fields now block freezing unless the user explicitly treats the value as missing or leaves the field unmapped. Inspection reports their presence. Literal CSV strings such as `#N/A` retain the user's explicit missing-value policy. Existing formula and numeric-identifier checks remain in place, and the original workbook bytes are retained in the frozen dataset.
- **Late uploads overwriting newer choices.** An earlier file read or failure cannot replace a newer upload, server path, loaded draft, or unmounted editor. Oversized uploads fail before their bytes are read.
- **Re-reading metadata resetting mappings.** Reinspection of the same source preserves selected and removed attributes, renamed fields, identity columns, ownership, types, categories, and missing-value rules. A selected field that disappears remains visible for validation instead of being silently discarded. A different source receives fresh suggestions.

The audit also reviewed existing duplicate slide-ID and physical-alias protections and added regression coverage. Discovery already rejects paths sharing the same device/inode, including hard links registered under different slide and patient IDs. This is a verified existing protection, not a newly introduced fix.

Unresolved patient identity remains explicit. The separately acknowledged slide-ID fallback is not presented as verified patient identity: two slides from one unknown patient could still enter different groups.

## Targets, eligibility, and split isolation

Three leakage or split-definition defects now block preview and publication:

1. **Imported partition/fold aliases used as predictors.** Checks follow the dataset dictionary's source-column mapping, including older protocol versions. Renaming an ordinary source column such as `assignment_code` no longer bypasses the exclusion.
2. **Identity or assignment source columns recoded into an apparently valid target.** Checks now follow reviewed slide/patient identity source columns and imported partition/fold source columns, blocking aliases even when their class mapping appears valid and balanced. This closes source-mapping bypasses across protocol versions.
3. **An excluded predefined validation cohort silently replaced by generated validation.** If eligibility or target missing/unmapped policies remove every originally designated validation slide, the affected imported-split workflow now reports `EMPTY_IMPORTED_VALIDATION`. The user must restore eligible validation groups or explicitly revise the partition mapping. It cannot silently sample replacement validation patients from training.

Existing checks against direct identity targets, direct target aliases, and one-to-one target-equivalent predictors were reviewed and regression-tested. The existing held-out-domain check also prevents using that domain as the prediction target. These protections were not all introduced by this audit.

Binary targets require an explicit positive class. The backend already enforced this requirement; the UI no longer chooses the first source label automatically. Missing and unknown positive-class selections are regression-tested. Explicit valid choices survive subsequent editing.

The split audit checked patient grouping, cross-slide rule overlap, explicit training/validation/test pools, class feasibility, imported assignments, nested folds, and multiple seeds. Added unstratified checks cover all five current CV strategies with seeds `0`, `7`, and `4294967295`. Invalid class coverage is blocked instead of being described as a usable split. These are membership and configuration checks, not model-performance tests.

### Independent browser fixture

The browser created and froze a real protocol using **48 slides from 24 patients, two slides per patient**. It reserved **8 external-test patients** and ran a **5-fold CV definition within the other 16 patients**, using seed `42`. An independent script read the saved memberships and checked all six plans:

| Plan | Training patients | Validation patients | Test patients |
| --- | ---: | ---: | ---: |
| Final | 14 | 2 | 8 external |
| Fold 0 | 10 | 2 | 4 |
| Folds 1–4, each | 11 | 2 | 3 |

All assertions passed: patient roles are disjoint within each plan; both slides from each patient stay together; each of the 16 training-pool patients appears in the CV test role exactly once; all 8 external-test patients are absent from every CV plan; the final test set is exactly the reserved external cohort. CV test membership is the out-of-fold assessment membership, distinct from the external final test set.

Browser filter checks also passed: selecting the negative outcome showed **24/48 slides**; adding the positive outcome with OR restored **48/48**; adding Site A with AND reduced the selection to **24/48 slides from 12 patients**. This verified that additional OR choices remain available while other filters still apply.

## Review, freeze, and recovery

Preview remains a review of specific inputs and a specific draft revision. New publication attempts recheck freshness and optimistic concurrency before freezing. When a review becomes stale or another editor changes the draft, the UI clears the obsolete review and provides **Reload saved draft** in both dataset and protocol editors. Reload fetches the latest server revision and discards unsaved local edits; the version tag and note survive when reloading the same draft.

The browser reopened a dataset for revision and verified that re-reading preserved attribute types and owners. After removing Site and re-reading again, the DOM contained exactly the selected attributes **Outcome, Age, and Partition**. The revised draft was saved at revision 1, then independently changed through the API to revision 2. The older UI's preview was rejected because the import draft had changed. **Reload saved draft** recovered revision 2 and its new name, leaving zero alerts.

Completed immutable publication retries are handled separately from new publications. A repeated successful operation returns its original frozen feature registration, bundle, or protocol without rebuilding scientific content from subsequently changed sources. Reusing the operation ID for a different request still conflicts. Current source/pack staleness is reported separately and does not rewrite the historical artifact. Protocol regression checks retain the existing expected preview hash for unchanged valid inputs.

Invalid import/protocol specifications now report concise field paths and validation messages, without exposing raw input dictionaries or Pydantic diagnostic URLs. Viewing a saved protocol shows its saved detail; copying it reopens the editable draft workflow.

## Features, extraction provenance, and packs

Feature acquisition and immutable bundles belong to Features. The bundle records the exact feature configuration and zero or more explicitly verified packs. A features-only bundle is valid. Loading policy belongs to MIL experiment planning, rather than becoming a mutable preference on the feature artifact.

The feature audit fixed the following integrity gaps:

- Source or extraction evidence changes cannot slip through a new feature publication at the publication boundary. Same-size replacements and restored modification times are covered by the broader filesystem identity checks.
- Extraction completion requires explicit, consistent coverage and completed inspection without blocking findings. A successful process exit, incomplete receipt, missing outputs, or unvalidated slides cannot independently establish success.
- Linked extraction evidence must match the dataset, feature output folder, feature kind, encoder, and requested task. Flat-layout encoder inference is checked against the linked extraction. Externally supplied features remain attachable without inventing extraction provenance.
- Embedded feature coordinates must equal the upstream requested patch coordinates, including feature-only jobs. Matching dimensions or patch counts alone are insufficient. Pooled slide embeddings retain their distinct shape rules.
- Changes to source slides or explicit checkpoint files during extraction invalidate verified completion. Replacing a checkpoint at the same path prevents incompatible reuse of existing embeddings.
- Extraction outputs cannot be placed inside managed metadata or an existing immutable pack.

Bundle freezing requires current full source-tensor validation and current full verification of every included pack. Verification checks slide membership, per-slide and total patch counts, dimensions, dtype, offsets, payload lengths, finite values, every feature value, and normalized coordinates against the source. Header compatibility alone is not called verification. HistoPilot packs require their manifest and checksum file together; losing one cannot silently downgrade the artifact into the legacy four-file format. Genuine legacy packs remain supported through source comparison. Precision conversion, when explicitly selected, is recorded rather than represented as exact original precision.

The browser froze **“Audit features only”** and **“Audit features with pack.”** The actual packing worker processed all 48 HDF5 files into **768 patches × 8 float32 features**: `features.bin` contains **24,576 bytes**, and `coords.bin` contains **6,144 bytes** of int32 XY coordinates. Independent NumPy, PyArrow, and HDF5 reads confirmed that **every packed feature and coordinate row equals its source row**.

MIL Auto resolved both original-source loading and packed mmap loading without blocking findings. The browser then saved an MIL plan at revision 1. This confirms input planning and persistence, not training execution.

## Validation and evidence

| Check | Result |
| --- | --- |
| Full backend suite | **919 tests passed** |
| Full frontend suite | **115 tests passed** |
| TypeScript, production frontend build, and web bundling | Passed |
| Ruff and diff whitespace checks | Passed |
| Independent saved protocol memberships | Passed, six plans |
| Actual pack worker and independent full payload comparison | Passed, all 48 slides |
| Features-only and packed bundle freezing in browser | Passed |
| MIL Auto original-source/mmap resolution and saved plan | Passed; saved revision 1 |
| Browser OR/AND filter counts and patient totals | Passed |
| Browser re-read mapping preservation and stale-draft recovery | Passed for dataset and protocol; recovered revision 2 and valid protocol preflight |

The build retains the existing large-chunk warning; dependency deprecation warnings also remain. Neither was reported as a new failure from this audit.

The [saved audit evidence](evidence/dataset-target-feature-audit-2026-09-10.json) records the test results, independent assertions, and browser checks. No unexpected JavaScript errors were reported. Saved protocol controls and the dataset editor were also checked at desktop and mobile widths.

The isolated run is at `/tmp/histopilot-workflow-audit-20260910-tjxsgg99`. Its `split-audit-evidence.json` and `pack-audit-evidence.json` contain the independent assertions summarized above; `mil-plan.png` records the saved-plan browser result. This is temporary local evidence, not a durable repository artifact.

Relevant regression coverage is in `tests/test_imports.py`, `tests/test_protocols.py`, `tests/test_cv_strategies.py`, `tests/test_explicit_split_pools.py`, the feature/extraction/bundle/pack test modules, and `web/src/lib/datasetImport.test.ts`.

## Remaining limits

- **Recorded IDs are not proof of biological independence.** The software cannot determine whether different patient IDs refer to the same person, or whether distinct WSI files are copies or related sections. Device/inode checks catch physical aliases, not copied file contents or undocumented cross-cohort overlap.
- **Filesystem freshness is not authenticated provenance.** Size, timestamps, device, inode, and verification receipts detect ordinary changes. They do not authenticate the origin of WSIs, checkpoints, or upstream extraction claims. Pack payload comparisons/checksums provide narrower integrity evidence; they do not establish the source data's scientific correctness.
- **External PFM pretraining overlap is unknown.** The audit cannot establish that pretrained encoders have never seen evaluation patients or related data. Nor can alias checks discover every clinically derived target proxy, post-outcome variable, or undocumented label transformation.
- **Training was not executed.** No GPU extraction or MIL training run was available for this audit. Membership isolation and input planning do not verify a future trainer's fitted preprocessing, sampling, checkpoint selection, metrics, or external-test discipline.
- **The browser evidence is synthetic and bounded.** It exercises actual persistence and packing with known expected results. It is not a new production-scale benchmark, a scan of all user data, or proof that every UI state and external dependency behaves correctly.

Before interpreting a trained model's results, the experiment still needs reviewed patient identities and target provenance, training-only fitted transforms and model selection, and an execution implementation that honors the frozen memberships and feature inputs.
