# Scientific workflow review — HistoPilot-dev

Reviewed 2026-09-12 against the current source, including import, protocol generation,
feature binding, development planning, training, predictor publication, refit,
evaluation, and clinical reporting. This is a source and regression review; it does
not establish clinical validity or model performance on a real cohort.

## Verdict

HistoPilot has a strong scientific foundation for its implemented image-feature
ABMIL and development-only k-fold workflow. Its most valuable properties are exact
frozen memberships, explicit label encoding, grouped splitting, separately held
early-stopping data, checkpoint provenance, and a distinct external evaluation
stage. These should be preserved. Rewriting the pipeline would discard useful,
already-tested safeguards.

The main weaknesses found in this review were boundaries where a declared choice
could acquire a different meaning downstream: optional spreadsheet predictors were
silently ignored, unverified slide fallback groups became apparent patient
identities in clinical reports, and incompatible evaluation choices could be frozen
before execution rejected them. Source-file aliases also bypassed ID-only overlap
checks across separate imports. The dev changes below address those concrete gaps.

The product remains a research workflow with a narrower executable scope than its
study-design controls. Arbitrary split strategies, multimodal models, clinical
validation, and unlimited-scale execution are not established by these changes.

## Findings implemented in this branch

| Priority | Before | Dev behavior | Evidence |
|---|---|---|---|
| High | The protocol UI offered extra spreadsheet columns “for the model,” but the ABMIL loader/model consumed image features alone. A nonempty `spec.predictors` passed MIL input review. | MIL input review returns `TABULAR_PREDICTORS_UNSUPPORTED`. The user must choose a protocol revision without extra spreadsheet inputs. Existing protocols remain inspectable. Training launch already reuses this review, so historical plans cannot newly launch while silently omitting inputs. | `histopilot/application/mil_inputs.py`, `tests/test_mil_inputs.py::test_extra_spreadsheet_predictors_cannot_be_silently_ignored_by_image_only_training` |
| High | Independent imports could assign different slide and patient IDs to the same WSI and pass external-cohort overlap review. Imports already rejected physical aliases within one inventory, but cohort review compared ID strings only. | Cohort review also compares canonical source paths and available frozen `(device, inode, size, mtime)` evidence. It blocks renamed or hardlinked development slides with `DEVELOPMENT_SLIDE_SOURCE_OVERLAP`. Only actual development members participate; an unused slide from the same dataset remains eligible. No live WSI scan or pixel read is introduced. | `histopilot/application/evaluations.py::_slide_sources`, `tests/test_evaluations.py::test_cross_import_slide_source_aliases_cannot_bypass_development_overlap` |
| High | Inference dropped `patientIdSource`. Clinical reports treated acknowledged `slide_fallback` IDs as verified patients and could produce independence-based Wilson intervals. | New prediction JSON preserves identity source. Clinical review obtains the authoritative source from frozen cohort membership, including for old prediction files that omitted it. Contradictory prediction metadata is rejected. Patient clinical analysis requires verified IDs; slide analysis remains available with explicit warnings and no independence intervals for fallback groups. Patient counts exclude fallback groups. | `histopilot/training/inference.py`, `histopilot/application/clinical.py`, direct and service-level regressions in `tests/test_clinical.py` |
| Medium | The cohort form/schema allowed maximum patient aggregation, while frozen predictors and evaluation execution require mean probabilities. The cohort could be frozen into an unusable state. | Cohort preview reports `PATIENT_AGGREGATION_UNSUPPORTED` before freezing. Existing maximum-aggregation drafts can still be read and corrected. | `histopilot/application/evaluations.py`, `tests/test_evaluations.py::test_cohort_blocks_unsupported_patient_aggregation_before_freezing` |
| Medium | Cohort review checked encoder and dimensions but omitted dtype, despite evaluation execution enforcing an exact dtype contract. | Cohort preview reports `FEATURE_DTYPE_MISMATCH` before freezing. This aligns stages; it does not claim that every numerical dtype conversion is scientifically invalid. | `tests/test_evaluations.py::test_cohort_blocks_dtype_mismatch_before_freezing` |
| Medium | `StorageError` inherits `ValueError`; the broad clinical parsing exception handler replaced actionable clinical eligibility failures with a generic “malformed” error. | `StorageError` is preserved before the parser exception handler, so the UI receives the actual reason and remediation. | `histopilot/application/clinical.py::ClinicalService._prepare`, the legacy fallback service regression |
| Medium | An independent patient namespace warning was visible while preparing a cohort but disappeared from the downstream clinical report. | The report carries a warning that cross-dataset patient overlap could not be verified. This does not change the user's declared namespace choice. | `tests/test_clinical.py::test_clinical_report_preserves_unverifiable_cross_dataset_patient_overlap` |

## Existing strengths verified by source inspection

### Import and identity

`application/imports.py` preserves explicit identifier values, reports numeric
spreadsheet identifiers and potentially lost formatting, rejects duplicate main
keys and duplicate crosswalk keys, detects conflicting patient links and patient
attributes, and requires an explicit fallback policy. Slide discovery is bounded,
captures canonical paths and file identity, and rejects aliases within one import.
An absent patient mapping is not silently turned into a real patient.

The dataset materializes source tables, dictionary, records, provenance, inventory,
and exclusions. This makes later checks reproducible without rewriting user data.
Tests in `test_imports.py`, `test_protocols.py`, and `test_development_protocols.py`
cover relevant identity and selection boundaries.

### Target and splitting

Targets freeze class order, raw-label mapping, prediction unit, positive class, and
missing/unmapped-label policy. Protocol review checks identifier aliases and target,
partition, and domain leakage. Rule conjunctions evaluate on the same slide, then
expand selections to the entire patient group. Conflicting group assignments block
publication. Selection of the development population happens before requiring
labels on unrelated rows in a combined development/test table.

Modern split generation deterministically hashes seed/context/group identities,
preserves patient groups, warns when minimum group constraints change requested
fractions, and freezes exact plan memberships. Nested inner plans exclude outer
assessment patients. Leave-one-domain-out excludes validation groups from the
held-out domain. These design properties have focused tests in
`test_cv_strategies.py` and `test_development_protocols.py`.

### Features and development execution

MIL input review requires compatible frozen protocol/bundle identity and exact
feature coverage. A selected pack must belong to its bundle. Automatic loading
does not silently choose among several packs or accept an undeclared change of
precision. Legacy protocols with pinned packs retain their representation binding.

Execution plans retain exact memberships and source stamps. The dataset layer
validates class order, patient roles, feature shape and finite values. Patch and
slide-order sampling depend on frozen seed, epoch, and identity. Epoch travels with
sampled indices, which avoids worker-prefetch ambiguity on resume. Patient-target
training assigns inverse-slide-count weights so patients with many slides do not
automatically dominate the objective.

`training/fold.py` uses validation metrics alone for checkpoint selection and opens
assessment data after fitting. A verified `fit-complete.json` permits resuming
interrupted assessment without another training epoch. Checkpointed RNG state,
optimizer/scheduler state, finite-gradient checks, and resource teardown are
present. Relevant tests include assessment-label invariance, patient leakage,
deterministic resume, minimum-epoch early stopping, and worker shutdown in
`test_mil_training.py`.

### Predictor, refit, and external inference

Predictor publication checks complete candidate/seed fold membership and verifies
frozen plan, batch, protocol, feature, run receipt, and checkpoint evidence. A live
or incomplete source run cannot simply be treated as a finished model. Refit derives
its fixed epoch budget from best checkpoint epochs, rather than stopped epochs,
and includes each selected development slide once. External/final memberships are
rejected. The refit loader has no validation or test loader.

Inference rechecks features and checkpoint hashes, retains exact slide order and
membership, uses whole bags, and averages ensemble probabilities. Member caches
have input and content digests. Binary decisions respect the frozen positive class
and configured threshold; ranking metrics use probabilities. Unlabeled rows are
retained in predictions and excluded from metric denominators.

### Metrics and clinical reports

The metric implementation checks class encoding and finite normalized probability
vectors, handles ties explicitly for ROC AUC and average precision, and returns
undefined ranking metrics when the required class is absent. Macro F1 uses all
frozen classes; balanced accuracy uses classes represented in the current cohort.
These are distinct definitions and should remain visible in comparisons. Stable
log probabilities preserve loss for extreme finite logits.

Clinical reports recompute descriptive quantities from receipt-verified stored
prediction evidence and validate cohort memberships, labels, target identity,
patient slide lists, and mean aggregation. They label threshold overrides as
descriptive and report assumptions for decision curves. Correlated-slide intervals
were already suppressed; the dev fix extends that protection to fallback IDs.

## Remaining limitations and recommended next work

1. **Executable scope is k-fold ABMIL.** Other split strategies can express study
   designs, but development planning blocks their execution. Nested CV needs
   per-outer-fold search, inner selection, and selected outer refit orchestration.
   A UI control should distinguish design-only support from an executable choice.
2. **Patient independence still needs correct source data.** Fallback grouping is
   explicitly acknowledged and continues to support exploratory development, but
   cannot establish patient-independent assessment. Distinct identifier namespaces
   cannot prove that two datasets contain different people. The software cannot
   infer a crosswalk that the source data do not provide.
3. **Source alias detection has a defined boundary.** Canonical paths and retained
   file identity detect common local aliases. Copies with different paths/inodes,
   rescanned tissue, transformed duplicate images, feature-only imports without WSI
   provenance, and inventories from different machines need additional content or
   specimen-level identity evidence. The review does not claim those are detected.
4. **External test results can influence later human choices.** The system permits
   evaluating several predictors or thresholds on one cohort. Freezing inputs
   records those choices but does not by itself preregister a primary model or stop
   adaptive reuse of the test cohort. A future study-selection record should state
   the primary model, cohort, operating point, and exploratory comparisons before
   opening results.
5. **Clinical utility is descriptive.** Existing intervals condition on fitted
   models and chosen thresholds; AUC/Brier uncertainty, paired comparisons, and
   full training/selection uncertainty are not estimated. Cohort representativeness
   and downstream clinical benefit are not established by these calculations.
6. **Feature compatibility is stronger when extraction provenance exists.** Current
   evaluation checks explicit encoder, dimension, dtype, and available extraction
   settings. Manually supplied matching encoder names do not prove identical
   checkpoints, preprocessing, magnification, or tissue quality. Pack precision can
   be explicitly changed; performance effects still need experiment-specific review.
7. **Performance needs representative benchmarks.** Whole-bag inference pads batches
   to the longest slide and repeated source/checkpoint verification has I/O cost.
   No large-WSI, GPU-memory, multi-user, or multi-day benchmark was run in this review.

## Compatibility and validation

Valid existing evaluation cohorts keep their scientific preview-hash shape. New
compatibility checks produce findings without inserting new fields into valid
cohorts; alias evidence is added only to a blocked preview. Cohorts that were
previously accepted despite an unsupported choice or source alias now fail review.
Existing saved clinical reports remain immutable. New report generation obtains
identity source from the frozen cohort, so missing metadata in older predictions
does not bypass the fix. No dataset, model, checkpoint, or existing result is rewritten.

Focused verification completed:

- `.venv/bin/python -m pytest tests/test_clinical.py tests/test_evaluations.py -q -k 'not api'`
  — **84 passed, 1 deselected**.
- `.venv/bin/python -m pytest tests/test_mil_inputs.py -q -k 'not api'`
  — **41 passed, 1 deselected**.
- Ruff check and format over the eight changed source/test files — **passed**.
- The inference provenance regression was added to the existing real-inference
  fixture test. The review environment's `.venv` has no PyTorch, so that module was
  **skipped**, not represented as runtime-verified.

API TestClient coverage was excluded from these focused commands after an in-sandbox
TestClient hang. The parent integration review handles the complete suite outside
that sandbox boundary and records its result separately. No HistoPilot server was
started or restarted, and no real training, extraction, or evaluation job was launched.
