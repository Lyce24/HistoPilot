# Development targets and split strategies

Stage 2 defines **development data only**: the training cohort, target labels, eligibility, patient grouping, and reproducible development splits. New protocols use `split.version: 4`. They contain no reserved external cohort and create no final evaluation plan.

Configure inference cohorts in **Model evaluation**, while experiments are running or after models have been developed. A shared metadata file with column values or conditions identifying different cohorts is recommended. Separate imported files are also supported there; evaluation setup checks their slide IDs against the selected feature representation and packs.

## Select development sources

Start with a frozen dataset and select eligible records using cohort conditions. Within that selection, use **all remaining eligible slides**, custom training conditions, or a predefined source column. A matching slide selects its entire eligible patient group.

Predefined values can map to `train`, `trainval` (training), or `val`. Unmapped values and rows outside custom training/validation conditions remain outside the development protocol. These rows are not automatically reserved for any later evaluation. Targets and feature coverage are checked only for the selected development records. Target suggestions and value counts use this same selected population.

The **Early-stop validation source** defaults to **Sample a percentage of training**, initially 15%. Choose **Use my fixed validation set** to preserve an official validation partition or select validation groups with conditions. Fixed validation groups stay outside fitting and assessment folds. An imported validation partition is never silently added to training: choose fixed validation or explicitly remap those values.

Overlapping training/validation groups, empty required sets, and missing selected labels block freezing. Unselected records do not need labels for the development target.

## Choose a development strategy

| Strategy | Development assignments |
| --- | --- |
| K-fold | Rotate assessment folds; sample early-stop validation from the remaining fitting groups, unless fixed validation is selected |
| Monte Carlo | Independently sample assessment groups per repeat; obtain early-stop validation from the remaining fitting groups or fixed source |
| Leave-one-site/cohort-out | Hold out each selected development site/cohort for assessment; early-stop validation uses fitting domains only |
| Nested K-fold | Rotate outer assessment folds, inner configuration-selection folds, and separate early-stop validation |
| Development holdout | Create one development assessment subset per split seed, separate from fitting and early-stop validation |

These strategies compare configurations during development. Training seeds, learning rates, weight decay, maximum epochs, stopping metrics, and search settings belong to **Model development**. Split seeds here control cohort assignments; training seeds later control training randomness without changing those assignments.

A fixed validation group belonging to a held-out site/cohort is omitted from that plan, and the omission is recorded. Insufficient validation data from the remaining domains blocks the plan.

## Percentage meaning

The early-stop percentage applies after development assessment and inner tuning groups are removed. With 100 development training groups, a five-fold plan contains approximately 20 assessment groups, 12 early-stop validation groups (15% of the other 80), and 68 fitting groups.

A development holdout with a 20% assessment fraction uses the same approximate allocation. Fixed validation overrides automatic early-stop sampling. Patient groups are indivisible; preview shows actual counts and warns when minimum-group constraints change requested allocations beyond ordinary rounding.

## Target defaults

Target attribute, task, class names, and positive class start empty. Choosing an attribute suggests class names and an identity label mapping from distinct nonmissing values in the selected development records. Two values suggest binary classification; more than two suggest multiclass classification. **The positive class always requires an explicit user choice.** Value order and frequency do not identify the clinical positive outcome.

Users can edit suggestions. Empty, single-class, or truncated value lists require explicit configuration. Changing development selection updates the displayed source values without silently overwriting an already reviewed target mapping. Loading a saved draft preserves its settings.

## Nested CV roles

Each outer fold creates inner tuning plans and one outer assessment/refit plan:

1. Hold out an outer assessment fold from the selected development training groups.
2. Rotate inner tuning folds within the remaining development groups.
3. In each inner plan, reserve separate early-stop validation. The configuration-selection fold is recorded as `tune`, separately from `val`.
4. Select settings using inner results, refit using the outer development data, and assess on the untouched outer fold.

Outer assessment groups are absent from every inner plan. Inner tuning scores and outer assessment scores are distinct outputs. Freezing saves these assignments; it does not train, choose a configuration, or generate predictions.

## Grouping, coverage, and interpretation

All eligible slides belonging to a supplied Patient_ID stay in one role within each plan. Confirmed Slide_ID fallback groups remain visibly separate from verified patient identities. Slide IDs alone do not establish patient independence.

Stratification is enabled by default. Whole-site assessment folds retain their observed label composition; the service does not move patients between sites to balance them. Missing or inconsistent domain values block site/cohort splitting. A domain assessment fold missing a class produces a warning because some metrics cannot be computed for that fold.

Preview checks group overlap, label mappings, minimum group/class support, and bounded assignment sizes. The frozen protocol records split seeds, fold levels, held-out domains, exact memberships, and algorithm identity.

Planned OOF coverage reports how often development groups appear in assessment folds. It does not mean predictions have been generated. K-fold and complete outer folds provide one assessment appearance per group per split seed; Monte Carlo repeats may assess a group multiple times or never assess it. Source code retains the historical internal partition key `test` for these assessment assignments. In version 4, this key means **development assessment**, not an independent final cohort. Each plan carries `pool: development`, and the summary explicitly records `scope: development`.

OOF scores used to choose a configuration are development evidence. An independent inference cohort or correctly nested outer assessment is needed to assess the selected procedure without reusing the configuration-selection evidence.

## Saved protocols and compatibility

Versions 1–3 retain their original assignments, interpretation, serialization, preview hashes, and freeze replay behavior. Version 3 explicitly reserved training, optional fixed validation, and final-test source pools and generated final plans. Version 2 assessed the eligible cohort using its original strategy semantics. Version 1 retains its older rules and train/validation split behavior.

Legacy designs remain readable. **Create a development-only draft** starts a new version-4 design and requires reviewing its training selection. Existing frozen records are never migrated in place. Inference cohorts are separate versioned artifacts that reference the selected development protocol.

## Verification

Focused tests cover all five development strategies, sampled and fixed validation, combined-file filtering before label validation, predefined training values, patient overlap, rejection of reserved test pools in version 4, legacy frozen-record preservation, and development-only labels in the rendered controls. Legacy strategy and protocol tests continue to verify their original behavior.
