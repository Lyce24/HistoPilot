# Methods

This page defines how HistoPilot assigns slides to roles, trains and selects models, and computes the numbers it reports. Read it before interpreting a result or describing a HistoPilot analysis in a paper. Every definition comes from the current code; the module that implements each part is named in its section.

## Populations and roles

A study passes through three levels of membership. Each is frozen before the next is designed.

| Level | Set by | Roles |
| --- | --- | --- |
| Targets & splits | A frozen target/split version | **Training** set and **testing** set. The testing set becomes a test cohort. |
| Cross-validation | Experimental Setup, inside the training set only | Per fold: **train** (fitting), **val** (early-stop validation) and **test** (the development *assessment* fold) |
| External cohorts | Test cohorts | Any number of separately prepared labeled or unlabeled cohorts |

The testing set of a target/split version never enters fitting, early stopping, configuration selection or refitting. Development split seeds redraw only the folds; they never redraw the testing set. Inside a fold plan the assessment fold is stored under the key `test`, with `pool: development`; it is a held-out development fold, not an independent test cohort.

## Split units and grouping

`histopilot/application/target_splits.py`, `application/protocols.py`

The **split unit** decides what is assigned to training or testing:

| Unit | What is assigned | Scoring unit |
| --- | --- | --- |
| Slide | Each slide on its own | Slide |
| Patient | Each patient's group of slides, kept together | Patient (slide probabilities are aggregated first) |

The UI starts new target/splits at **Slide**. Versions saved without a split unit, which includes every version created before the option existed, are patient-grouped. The target's unit must equal the split unit.

Under patient grouping, slides are grouped by their patient ID. A missing patient ID blocks the version unless the dataset explicitly chose **Slide-ID fallback**, which makes each such slide its own group and records `patientIdSource: slide_fallback`. Fallback groups are reported separately from verified patients: they do not establish that two sets are patient-disjoint, and patient-level scoring and patient bootstrap intervals refuse them.

**Slide-unit designs do not keep a patient's slides together.** Two slides of one patient can land in training and testing. For folds only, Experimental Setup offers **Keep all slides of a case in the same fold** (`groupByPatient`): folds and early-stop validation then keep each patient's slides together, while labels and scoring stay per slide. The testing set is unaffected by this option. Use the patient unit when patient independence between training and testing matters.

### Assigning training and testing

- **Random split.** Units are ordered by a SHA-256 hash of the algorithm name, split seed, stratum and unit ID, separately within each stratum. Stratification is off by default; when on, it uses a metadata field you choose, never the target, and a patient whose slides disagree on that field forms its own stratum. Each stratum sends `round(n × testing fraction)` units to testing, rounding halves up. When the fraction is above 0 and a stratum has at least two units, that count is kept between 1 and n − 1, so every such stratum contributes to both sets. A one-unit stratum goes to testing only at 50% or more. With many small strata the overall testing share can differ from the request; review shows the actual counts.
- **Metadata conditions.** Testing conditions select testing units. Empty training conditions take the remaining eligible units; explicit training conditions take a subset and exclude the rest. **Use all eligible … outside the training set** makes testing take everything training does not.
- **Predefined partition values.** Values of one metadata column map to Training, Testing or Exclude.

A unit matching both sets, a patient whose slides fall in both sets, or one physical slide file in both sets blocks freezing. The training set must be non-empty and contain every target class. A binary target always needs an explicit positive class.

## Cross-validation

`histopilot/application/modern_splits.py`, `application/development_splits.py`

Experimental Setup runs **k-fold** cross-validation inside the frozen training set; other strategies are no longer offered for new setups. Defaults are 5 folds (2–10 allowed), split seed 42 (up to 10 split seeds) and stratification on.

**Folds.** Groups (patients, or slides for a slide-unit design without `groupByPatient`) are stratified by target label. A patient whose slides carry different labels forms a stratum of its own label combination; HistoPilot never picks one slide's label for the patient. Within each stratum, groups are ordered by a SHA-256 hash of the seed and group, then dealt round-robin into the folds, starting at the currently smallest fold.

**Early-stop validation.** For each fold, the groups outside the assessment fold form the fitting pool. Early-stop validation takes a fraction of that pool, 15% by default, per label stratum: `round(n × fraction)` groups, halves up, adjusted so that each stratum keeps at least the minimum number of groups on both sides when it has enough groups. The remaining groups are fitted. For example, 100 training groups in five folds give about 20 assessment groups, 12 validation groups (15% of the other 80) and 68 fitting groups per fold. Every fold must have at least one group of each class in each role, or the design is blocked.

**Seeds.** A **split seed** fixes fold and validation membership. A **training seed** fixes model randomness (initialization, patch sampling and data order) without changing membership. A batch runs every configuration × training seed × split seed × fold; for example, 6 configurations × 3 training seeds × 5 folds with one split seed is 90 runs.

## Checkpoint and configuration selection

`histopilot/training/fold.py`, `histopilot/candidate_selection.py`, `histopilot/schemas/development.py`

**Checkpoints.** Each fold trains on its fitting groups and scores its validation groups after every epoch, at the target's unit (patient targets aggregate slide predictions first). The recipe's **checkpoint metric** is validation AUROC by default, or validation loss or accuracy. The best checkpoint is the first epoch that reaches the best value. Early stopping ends training after `patience` epochs without improvement (default 8, minimum improvement 0), but not before the minimum epochs (default 1) and never beyond the maximum (default 40). Records saved before these defaults existed keep their original settings (validation loss, 100 epochs, patience 15). Validation AUROC requires every class in each validation fold.

**Small validation folds.** A recipe can set a minimum number of positive validation units for a binary target, with a fixed epoch budget as a fallback. A fold below the minimum trains for exactly that budget without early stopping and keeps its final epoch. A fixed budget can also be set outright. nnMIL recipes can also choose to keep the latest epoch.

**Assessment.** Only after fitting is the assessment fold loaded and predicted by the selected checkpoint. Assessment outcomes never influence checkpoints, stopping or configuration choice.

**Configuration selection.** When a batch has several configurations, the reported one is chosen by validation, never by out-of-fold results. Each configuration's score is the mean, over all of its runs (folds × training seeds × split seeds), of the selection metric at each run's selected checkpoint. The **selection metric** is validation AUROC by default, or validation loss or accuracy. Loss is minimized and the others maximized; ties go to the lower configuration number. The choice is made only once every configuration has complete validation scores. Until then, Results show the lowest-numbered configuration with a warning.

Validation data drives both checkpointing and configuration choice, and out-of-fold results of the chosen configuration are still development evidence. Their intervals do not account for configuration selection. An independent test cohort is needed for an estimate that is free of these choices.

## Out-of-fold predictions

`histopilot/workers/train_batch.py`, `histopilot/scoring.py`

For each configuration, training seed and split seed, the assessment predictions of all folds are pooled into one out-of-fold (OOF) set. It is complete only when every fold finished, and it must cover each training slide exactly once with its frozen label; otherwise no OOF metrics are reported. Nothing is filled in for incomplete groups.

**Patient aggregation.** For patient targets, slide predictions are combined per patient by the recipe's rule: `mean_probabilities` (the default, the mean of slide probabilities) or `mean_logits` (the mean of slide log-probabilities, renormalized, which equals a softmax of mean logits). A patient's slides must share one label.

## Cross-validated results

`histopilot/application/experiment_results.py`, `histopilot/cv_summary.py`

The Results tab reports, for the selected configuration of each batch:

| Row | Definition |
| --- | --- |
| Per fold | Metrics each fold recorded on its own assessment fold, with its checkpoint epoch |
| Per seed | Metrics of one pooled OOF set: one training seed and one split seed, every training slide scored once by the fold model that did not train on it |
| Seed average | Mean ± sample SD (n − 1) of the per-seed OOF metrics across all complete training-seed × split-seed replicates, with minimum and maximum |
| Fold average | Mean ± SD of all per-fold metrics. Descriptive only: folds are small, and the pooled OOF value is the more stable estimate. |
| Seed ensemble | Each slide's predictions averaged across seeds (mean probability by default, or mean logit), then scored once. Needs at least two replicates. |

Headline numbers are seed-average OOF values; AUROC is the primary metric. The Results page also flags patterns without changing any number: a class with mean recall below 0.5, a seed SD of 0.02 or more in AUROC, a fold AUROC range of 0.10 or more, a median checkpoint epoch of 2 or less, a single training seed, and a missing validation-based choice.

## Metrics

`histopilot/training/module.py` and `histopilot/cv_summary.py`, which produce identical values. All are computed in NumPy.

| Metric | Definition |
| --- | --- |
| AUROC | Binary: the Mann–Whitney AUROC of the positive class, with tied scores counting one half. Multiclass: the unweighted mean of one-vs-rest AUROCs (macro), unavailable if any class lacks positives or negatives. |
| AUPRC | Average precision, Σ ΔTP × precision ÷ positives, with tied scores sharing one threshold (no interpolation). Multiclass: the unweighted mean over classes. |
| Balanced accuracy | Mean recall over the classes present in the labels |
| Macro-F1 | Mean F1 over all target classes. A class that is neither present nor predicted counts as F1 = 0, unlike some libraries' defaults. |
| Accuracy | Fraction of correct decisions |
| Loss | Mean negative log-likelihood of the true class (natural log). Reported, but never resampled or used to rank models. |

Rankings use one-vs-rest log odds computed from saved log-probabilities, which order predictions exactly like the probabilities but keep their order near 0 and 1. **Decisions** use argmax for multiclass targets. For binary targets a unit is positive when P(positive) ≥ the recipe's **decision threshold**, 0.5 by default. The threshold changes decisions and decision-based metrics, never AUROC or AUPRC. Metrics that are undefined for the data, for example when a class is absent, are reported as unavailable with the missing classes listed. Calibration, Brier score and log loss appear in [clinical utility](#clinical-utility).

## Confidence intervals

All intervals are 95% percentile bootstrap intervals with a fixed seed (42 by default) and 2,000 resamples by default (200–10,000). Draws are ordinary resamples with replacement, not stratified by class. A draw that lacks a required class is excluded and counted; an interval is reported only when at least max(100, 80% of the resamples) draws are valid. **Intervals condition on the fitted models.** They do not include the variability of retraining, configuration selection or threshold choice.

### Intervals for cross-validated results

`histopilot/cv_summary.py`

- **Unit.** Slides in a slide-unit design, patients otherwise. Resampling patients resamples all of their slides together and requires verified patient IDs.
- **Seed average.** Each draw scores every seed's OOF set on the same resampled units, then averages across seeds. The interval is for the seed mean. The seed ensemble gets its own interval on the same draws.
- **Metrics.** AUROC, AUPRC, balanced accuracy, macro-F1 and accuracy.
- **Paired batch comparisons.** Batches of one experiment share their folds. Two batches' selected configurations are compared as left − right: the difference of seed means, with an interval from the per-draw differences on shared draws. Pairing requires both to have scored the same units.
- **Fold-paired differences.** For each fold both batches completed, the difference of their seed means. Reported as mean ± SD, range, and counts of folds where each side is better. No interval or p-value is computed.

With `groupByPatient` in a slide-unit design, the bootstrap still resamples slides, so the interval ignores the clustering of slides within patients.

### Intervals for evaluations

`histopilot/statistics.py`

Evaluations of patient-unit predictors report patient bootstrap intervals for AUROC and AUPRC. They need verified patient IDs, one prediction per patient and at least two labeled patients per class. A paired comparison of two evaluations on the same patients reuses each draw for both and reports left − right. A sensitivity analysis keeps one slide per patient, chosen by a seeded hash of the patient and slide IDs before any score or outcome is read, and repeats the analysis. Slide-unit evaluations report no bootstrap intervals.

## Predictors

`histopilot/application/predictors.py`, `application/refits.py`, `training/refit.py`, `training/inference.py`

A predictor is a frozen model built from one complete group: one configuration, training seed and split seed, after all of its folds finished. Each batch chooses **Ensemble**, **Refit**, **Both** or **Skip**, and which **configurations to build**: only the configuration chosen by validation (the default) or all of them. A batch of 15 configurations × 3 training seeds with one split seed that chooses Both therefore builds 3 ensembles and 3 refits of its selected configuration, or 45 of each for all configurations. The number of folds does not multiply predictors.

**Ensemble.** The group's fold checkpoints (each fold's selected epoch). Inference averages the members' class probabilities (default) or their log-probabilities, renormalized, per the recipe's ensemble aggregation. For patient targets, patient aggregation follows the ensemble.

**Refit.** One new model trained on all development slides of the group (the union of every fold's train, validation and assessment slides) with the same recipe and training seed. The testing set is never included. There is no validation data, so the epoch count is fixed in advance:

1. Take each fold's best checkpoint epoch (one-based).
2. Take the chosen percentile of those epochs with linear interpolation: P50 (median), P75, P90, P100 (maximum) or any value from 1 to 100.
3. Round up.

For fold best epochs `[2, 3, 5, 8]`, P50 gives 4 epochs and P75 gives 6. The refit trains exactly that many epochs with no early stopping. Warmup is shortened to fit the budget, a cosine schedule spans it, and a plateau schedule, which needs validation, becomes a constant learning rate. A fold whose best epoch cannot be verified from its receipt or validation history blocks the refit; the stopped epoch is never substituted.

**Frozen scoring rules.** A predictor carries its recipe's decision threshold and patient aggregation. Evaluations use them and refuse a different threshold or aggregation (`EVALUATION_THRESHOLD_MISMATCH`, `EVALUATION_AGGREGATION_MISMATCH`). HistoPilot never tunes a threshold on a test cohort.

## Evaluation and inference

`histopilot/application/evaluation_runs.py`, `histopilot/inference_summary.py`

An evaluation scores one predictor on one labeled test cohort with whole slide bags, the frozen class order and the frozen scoring rules. Metrics use labeled units only; unlabeled units still receive predictions.

A cohort that shares any slide, or any source slide file, with the predictor's development data is refused. In patient-grouped designs a cohort that shares patients with development is refused too. Slide-unit designs do not check or flag shared patients, so exclude them yourself when patient independence matters.

Results can compare an ensemble with a refit only within the same cohort and scoring unit, and only for matching groups. Choosing between strategies on a test cohort uses up that cohort's independence for the chosen strategy.

An inference run applies a predictor to an unlabeled cohort. It reads no labels and computes no metrics. The same development-overlap rules apply, with one exception: in a patient-grouped design whose target is slide-level, new slides from development patients are allowed and flagged in every view and export. Its descriptions of the predictions are:

| Quantity | Definition |
| --- | --- |
| Confidence | The probability of the predicted class |
| Margin | Distance from the decision boundary on a 0–1 scale. Binary: (p − t) ÷ (1 − t) above the threshold t, (t − p) ÷ t below it. Multiclass: top probability minus the runner-up. |
| Member agreement | For ensembles, how many fold members make the same decision, and the spread of their probabilities |
| Threshold sweep | Binary predicted-positive counts at thresholds 0.05 to 0.95 |
| Run agreement | Agreement and Cohen's κ between two runs on the same cohort |

None of these is an accuracy estimate.

## Clinical utility

`histopilot/application/clinical.py`

A clinical utility report reads the checksummed predictions of a completed evaluation for one unit (slide or patient) and one outcome class. For multiclass targets the chosen class is compared with all others. It never refits, recalibrates or tunes a threshold; the operating point is the evaluation's frozen threshold unless you enter a descriptive override, which is labeled as such. Unlabeled units are counted and excluded.

| Quantity | Definition |
| --- | --- |
| Brier score | Mean of (p − y)². The reference is prevalence × (1 − prevalence); Brier skill score = 1 − Brier ÷ reference. Multiclass also reports the Brier score summed over classes (0–2). |
| Log loss | One-vs-rest negative log-likelihood |
| ROC and PR | Exact curves with tied scores grouped. Plots and CSV exports keep at most 2,001 points; AUC and average precision use every prediction. |
| Calibration | Equal-width probability bins (10 by default, 2–50): count, mean predicted probability and observed rate. ECE = Σ (bin count ÷ n) × \|mean predicted − observed\|; MCE is the largest bin gap. Also observed ÷ expected. |
| Operating point | Counts, sensitivity, specificity, PPV, NPV, accuracy, balanced accuracy, F1 and likelihood ratios at the threshold |
| Net benefit | TP/N − FP/N × t/(1 − t), compared with treat-all, prevalence − (1 − prevalence) × t/(1 − t), and treat-none, 0 |
| Standardized net benefit | Net benefit ÷ prevalence |
| Interventions avoided | Net interventions avoided per 100, relative to treating everyone |
| Clinical impact | Per 100 units: classified high risk, true positives, false positives and missed positives |

The decision curve sweeps thresholds from 0.01 to 0.99 by default. Wilson 95% intervals are given for sensitivity, specificity, PPV and NPV only when units are independent: the unit is the patient, or every slide has its own verified patient. The report has no bootstrap intervals and no calibration slope. Net benefit describes this cohort and its prevalence; it is not an observed treatment effect.
