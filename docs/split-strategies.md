# Targets, training/testing membership and training folds

**Targets & splits** fixes the study population, prediction target and training/testing membership from a dataset. **Experimental Setup** then configures folds and validation inside the frozen training set. These are separate scientific decisions and separate immutable records.

## Fixed training and testing membership

Start with a frozen dataset and apply cohort conditions. Choose the split unit, assign training and testing populations and inspect their live counts, then define the training target field, label unit, classes and mappings. A binary target requires an explicit positive class. Testing labels can be omitted for inference.

### Split unit

`splitUnit` decides what is selected and assigned:

| Unit | Meaning |
| --- | --- |
| Slide | Each slide is selected and assigned on its own. Conditions match individual slides; targets, distributions and later folds use slides. Patient identities are not checked. |
| Patient | Matching one slide selects its eligible patient group, and every selected slide of a patient stays in one set. Targets, distributions and later folds use patient grouping. |

New drafts default to **Slide**. Specifications saved without `splitUnit` (all versions created before the field existed) are patient-grouped and keep their hashes; opening such a draft shows the original patient grouping until **Slide** is chosen explicitly. The target's prediction unit must match the split unit.

### Partition methods

Choose one fixed partition method:

| Method | Membership rule |
| --- | --- |
| Random split | A testing percentage and seed assign whole units (slides, or intact patient groups); optional stratification balances a selected raw metadata field. Remaining units train. |
| Metadata conditions | Testing conditions select testing units. Empty training conditions use the remaining eligible units; explicit training conditions select a subset and exclude unmatched units. With training conditions set, **Use all eligible … outside the training set** (`testRemaining`) makes testing take every eligible unit that training does not select; testing then has no conditions of its own. |
| Predefined partition values | Map exact metadata values to Training, Testing or Exclude. Unmapped units are excluded. |

A unit cannot occur in both sets. Under patient grouping, rules or values that put one patient's slides in both sets block freezing; under slide splitting, a slide matching both training and testing conditions blocks freezing. Exact physical slide overlap between training and testing also blocks freezing. The training set must be nonempty and contain every target class. Testing may be empty, with a warning; use 0% when supplying another evaluation cohort later.

### Random split rounding

Units are ordered by a hash of the split seed and unit, separately within each stratum. Without stratification the whole eligible cohort is one stratum. With stratification (off by default; it requires a chosen metadata field), a stratum is the set of that field's values within a unit, so a patient whose slides disagree forms its own stratum. Each stratum sends `round(n × testing fraction)` units to testing, rounding halves up. When the fraction is above 0 and a stratum has at least two units, that count is kept between 1 and n − 1, so every such stratum contributes to both sets; a one-unit stratum goes to testing only at 50% or more. With many small strata the overall testing share can therefore differ from the requested percentage. Review shows the actual counts.

### Freezing and the testing cohort

Features, fold counts, early-stop percentages, training seeds and hyperparameters are not required here. Review shows training/testing slide counts (and patient counts under patient grouping), label counts and exact assignments. Freezing retains those memberships and then derives a reusable evaluation cohort for labeled testing, or an inference cohort when testing has no target. The freeze response and the version's detail (`GET /target-splits/{id}`) report it as `testCohort` (`required`, `id`, `state`). If deriving the cohort fails after the version was published, the freeze still succeeds and returns `testCohortError`; the detail view then offers **Retry test cohort**. A version whose testing set has no derived cohort (for example one frozen before this behaviour) offers **Create test cohort** (`POST /target-splits/{id}/test-cohort`). Reading a version never creates the cohort. Adding features or changing a setup never redraws this partition.

## Feature compatibility

Slide features can be prepared alongside Targets & splits. **Experimental Setup → Inputs** selects a bundle and checks it against all frozen training members. Missing features block setup preparation without changing the training population. Evaluation separately checks coverage for its selected testing cohort.

## Training fold design in Experimental Setup

New setups currently execute **K-fold** training only. The other strategies remain disabled in the setup editor. Their saved protocol semantics are documented below for inspecting historical records; they are not additional executable choices in the new pipeline. The current editor samples early-stop validation from each fold’s fitting groups and does not offer a fixed validation cohort.

| Strategy | Availability in new setups | Development assignments |
| --- | --- | --- |
| K-fold | Available | Rotate assessment folds; sample early-stop validation from the remaining fitting groups |
| Monte Carlo | Disabled; historical reference | Independently sample assessment groups per repeat; obtain early-stop validation from the remaining fitting groups or fixed source |
| Leave-one-site/cohort-out | Disabled; historical reference | Hold out each selected development site/cohort for assessment; early-stop validation uses fitting domains only |
| Nested K-fold | Disabled; historical reference | Rotate outer assessment folds, inner configuration-selection folds, and separate early-stop validation |
| Development holdout | Disabled; historical reference | Create one development assessment subset per split seed, separate from fitting and early-stop validation |

These strategies compare configurations during development. Training seeds, learning rates, weight decay, maximum epochs, stopping metrics, and search settings belong to **Experimental Setup**. Split seeds here control cohort assignments; training seeds later control training randomness without changing those assignments.

In historical protocols, a fixed validation group belonging to a held-out site/cohort is omitted from that plan, and the omission is recorded. Insufficient validation data from the remaining domains blocks the plan.

## Percentage meaning

The early-stop percentage applies after development assessment and inner tuning groups are removed. With 100 development training groups, a five-fold plan contains approximately 20 assessment groups, 12 early-stop validation groups (15% of the other 80), and 68 fitting groups.

In historical protocols, a development holdout with a 20% assessment fraction uses the same approximate allocation, and fixed validation overrides automatic early-stop sampling. Patient groups are indivisible; preview shows actual counts and warns when minimum-group constraints change requested allocations beyond ordinary rounding.

## Target defaults

Target attribute, task, class names, and positive class start empty. Choosing an attribute suggests class names and an identity label mapping from distinct nonmissing values in the selected dataset cohort. Two values suggest binary classification; more than two suggest multiclass classification. **The positive class always requires an explicit user choice.** Value order and frequency do not identify the clinical positive outcome.

Users can edit suggestions. Empty, single-class, or truncated value lists require explicit configuration. Changing cohort selection updates the displayed source values without silently overwriting an already reviewed target mapping. Loading a saved draft preserves its settings.

## Historical nested CV roles

Each outer fold creates inner tuning plans and one outer assessment/refit plan:

1. Hold out an outer assessment fold from the selected development training groups.
2. Rotate inner tuning folds within the remaining development groups.
3. In each inner plan, reserve separate early-stop validation. The configuration-selection fold is recorded as `tune`, separately from `val`.
4. Select settings using inner results, refit using the outer development data, and assess on the untouched outer fold.

Outer assessment groups are absent from every inner plan. Inner tuning scores and outer assessment scores are distinct outputs. Freezing saves these assignments; it does not train, choose a configuration, or generate predictions.

## Grouping, coverage, and interpretation

For patient-grouped target/splits, all eligible slides belonging to a supplied Patient_ID stay in one role within each plan. Confirmed Slide_ID fallback groups remain visibly separate from verified patient identities. Slide IDs alone do not establish patient independence. For slide-unit target/splits, folds and early-stop validation assign slides independently, unless the setup chooses **Keep all slides of a case in the same fold** (`groupByPatient`), which keeps each case's slides in one fold and one side of early-stop validation while labels and scoring stay per slide.

Version-4 **slide targets** may have different labels on different slides from one patient. Each slide keeps its own target, and the patient remains indivisible. When stratification is enabled, patient groups are stratified by their observed set of slide labels; no majority or highest grade is substituted. Per-class group counts count a patient once in every class represented by their slides, so those counts can sum to more than the patient count. The preview identifies this policy as `histopilot-development-labelset-plans-v4` and warns that patient-grade metrics are unavailable for conflicting labels. Use uniform slide sampling for the slide-based objective; patient samplers still require consistent patient labels. Patient targets and saved legacy protocol versions retain their consistency requirements.

Stratification is enabled by default. Whole-site assessment folds retain their observed label composition; the service does not move patients between sites to balance them. Missing or inconsistent domain values block site/cohort splitting. A domain assessment fold missing a class produces a warning because some metrics cannot be computed for that fold.

Preview checks group overlap, label mappings, minimum group/class support, and bounded assignment sizes. The setup’s derived training protocol records split seeds, fold levels, held-out domains, exact memberships, and algorithm identity. The source target/split record remains unchanged.

Planned OOF coverage reports how often development groups appear in assessment folds. It does not mean predictions have been generated. K-fold and complete outer folds provide one assessment appearance per group per split seed; Monte Carlo repeats may assess a group multiple times or never assess it. Source code retains the historical internal partition key `test` for these assessment assignments. In version 4, this key means **development assessment**, not an independent final cohort. Each plan carries `pool: development`, and the summary explicitly records `scope: development`.

OOF scores used to choose a configuration are development evidence. An independent testing cohort or correctly nested outer assessment is needed to assess the selected procedure without reusing the configuration-selection evidence.

## Saved protocols and compatibility

Versions 1–3 retain their original assignments, interpretation, serialization, preview hashes, and freeze replay behavior. Version 3 explicitly reserved training, optional fixed validation, and final-test source pools and generated final plans. Version 2 assessed the eligible cohort using its original strategy semantics. Version 1 retains its older rules and train/validation split behavior.

Legacy designs remain readable through **Open legacy development protocols**, which the Targets & splits library shows only when the project has a historical protocol version or draft; protocols derived by Experimental Setup are not listed there. Existing frozen records are never migrated in place. New target/split records contain only the fixed training/testing partition; Experimental Setup derives a separate development protocol from their training membership.

## Verification

Target/split tests cover random, rule-based and predefined partitions, patient overlap, source-slide overlap, label mapping, training class coverage, zero-test designs, immutable publication and exact derived test cohorts. Setup tests verify that reserved testing members never enter training assignments. Existing strategy tests preserve legacy CV behavior. The isolated `web/scripts/verify-target-split-workflow.mjs` fixture checks target construction, recovery, all three selection controls, freeze retries and the Experimental Setup handoff without starting a server.

## Partition-first target review

The editor proceeds through Dataset & cohort → Training & Testing split → Prediction Targets → Review & Freeze. Dataset & cohort uses a separate eligibility-only preview and shows remaining cases without training/testing details. In the split step, Training appears before Testing and each filter has an adjacent count and progress bar. Partition exploration accepts an unfinished target and updates counts as filters change, distinguishing direct rule matches, patient-expanded matches and assigned cases. Under patient grouping, known patient groups stay together, including slides expanded from a direct rule match. Label mapping does not reassign patients or refill a partition after explicit label exclusions.

Training targets are required. Testing can share the training mapping, use compatible source labels, or omit its target for pure inference. Inference testing members retain null labels even when metadata contains outcomes; they are offered to Run inference and do not produce evaluation metrics. Frozen versions retain their original memberships.
