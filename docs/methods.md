# Methods

This page defines how HistoPilot assigns slides to roles, trains and selects models, and computes the numbers it reports. Read it before interpreting a result or describing a HistoPilot analysis in a paper. Every definition comes from the current code; the module that implements each part is named in its section.

## Populations and roles

A study passes through three levels of membership. Each is frozen before the next is designed.

| Level | Set by | Roles |
| --- | --- | --- |
| Targets & splits | A frozen target/split version | **Training** set and **testing** set. The testing set becomes a cohort in Apply models. |
| Cross-validation | An experiment's design, inside the training set only | Per fold: **train** (fitting), **val** (early-stop validation) and **test** (the development *assessment* fold) |
| External cohorts | Apply models → Cohorts | Any number of separately prepared labeled or unlabeled cohorts |

The testing set of a target/split version never enters fitting, early stopping, configuration selection or refitting. Development split seeds redraw only the folds; they never redraw the testing set. Inside a fold plan the assessment fold is stored under the key `test`, with `pool: development`; it is a held-out development fold, not an independent cohort.

## Split units and grouping

`histopilot/application/target_splits.py`, `application/protocols.py`

The **split unit** decides what is assigned to training or testing:

| Unit | What is assigned | Scoring unit |
| --- | --- | --- |
| Slide | Each slide on its own | Slide |
| Patient | Each patient's group of slides, kept together | Patient (slide probabilities are aggregated first) |

The UI starts new target/splits at **Slide**. Versions saved without a split unit, which includes every version created before the option existed, are patient-grouped. The target's unit must equal the split unit.

Under patient grouping, slides are grouped by their patient ID. A missing patient ID blocks the version unless the dataset explicitly chose **Slide-ID fallback**, which makes each such slide its own group and records `patientIdSource: slide_fallback`. Fallback groups are reported separately from verified patients: they do not establish that two sets are patient-disjoint, and patient-level scoring and patient bootstrap intervals refuse them.

**Slide-unit designs do not keep a patient's slides together.** Two slides of one patient can land in training and testing. For folds only, an experiment's design offers **Keep all slides of a case in the same fold** (`groupByPatient`): folds and early-stop validation then keep each patient's slides together, while labels and scoring stay per slide. The testing set is unaffected by this option. Use the patient unit when patient independence between training and testing matters.

### Assigning training and testing

- **Random split.** Units are ordered by a SHA-256 hash of the algorithm name, split seed, stratum and unit ID, separately within each stratum. Stratification is off by default; when on, it uses a metadata field you choose, never the target, and a patient whose slides disagree on that field forms its own stratum. Each stratum sends `round(n × testing fraction)` units to testing, rounding halves up. When the fraction is above 0 and a stratum has at least two units, that count is kept between 1 and n − 1, so every such stratum contributes to both sets. A one-unit stratum goes to testing only at 50% or more. With many small strata the overall testing share can differ from the request. Review reports the share reached, in units and as a percentage, and warns when it is more than 5 percentage points from the request.
- **Metadata conditions.** Testing conditions select testing units. Empty training conditions take the remaining eligible units; explicit training conditions take a subset and exclude the rest. **Use all eligible … outside the training set** makes testing take everything training does not.
- **Predefined partition values.** Values of one metadata column map to Training, Testing or Exclude.

A unit matching both sets, a patient whose slides fall in both sets, or one physical slide file in both sets blocks freezing. The training set must be non-empty and contain every target class. A binary target always needs an explicit positive class.

## Cross-validation

`histopilot/application/modern_splits.py`, `application/development_splits.py`

An experiment's design splits the frozen training set only. Four designs train, because each assesses a group at most once per split seed, so every seed's assessments pool into one out-of-fold set (`development_splits.training_split_issue`):

| Design | Assessment folds | Assessed | Split seeds |
| --- | --- | --- | --- |
| **K-fold** (default) | Generated: 5 folds, 2–10 allowed | Every group | Each seed deals new folds |
| **Predefined folds** | One per value of a dataset column, such as the folds of a published study, in numeric order ("2" before "10") | Every group | The folds repeat; only early-stop validation changes (`PREDEFINED_FOLDS_REUSED` warns) |
| **Leave one site/cohort out** | One per value of a site or cohort column, or only the selected values | Every group, or those of the selected sites | The folds repeat; only early-stop validation changes |
| **Held-out assessment** | One stratified share of the training set (20% by default) | That share | Exactly one seed; repeat it over training seeds |

**Monte Carlo** designs can assess a group in several plans of one seed, and **nested** designs need a search inside each outer fold, so both remain for planning only (`TRAINING_SPLIT_UNSUPPORTED`, `NESTED_SELECTION_REQUIRED`). Defaults are split seed 42 (up to 10 split seeds) and stratification on.

**Folds.** Groups (patients, or slides for a slide-unit design without `groupByPatient`) are stratified by target label. A patient whose slides carry different labels forms a stratum of its own label combination; HistoPilot never picks one slide's label for the patient. Within each stratum, groups are ordered by a SHA-256 hash of the seed and group, then dealt round-robin into the folds, starting at the currently smallest fold.

**Predefined and site folds.** Every group needs one value in the fold or site column: a missing value (`MISSING_FOLD`, `MISSING_DOMAIN`) or slides of one group with different values (`INCONSISTENT_GROUP_FOLD`, `INCONSISTENT_PATIENT_DOMAIN`) block the design, as do fewer than two values or more than 20 predefined folds (100 sites). A fold column may be named like a split column (`fold`, `split_id`), but it cannot be an identifier or the target. Each membership row keeps its imported value (`foldValue`), and the design's summary lists the value behind each fold number. A site or predefined fold that lacks a class is a warning, not a block: that fold's own metrics are partial, and the out-of-fold set still pools it. Training and early-stop validation must hold every class.

**Early-stop validation.** For each fold, the groups outside the assessment fold form the fitting pool. Early-stop validation takes a fraction of that pool, 15% by default, per label stratum: `round(n × fraction)` groups, halves up, adjusted so that each stratum keeps at least the minimum number of groups on both sides when it has enough groups. The remaining groups are fitted. For example, 100 training groups in five folds give about 20 assessment groups, 12 validation groups (15% of the other 80) and 68 fitting groups per fold. Every fold must have at least one group of each class in each role, or the design is blocked.

**Seeds.** A **split seed** fixes fold and validation membership. A **training seed** fixes model randomness (initialization, patch sampling and data order) without changing membership. A batch runs every configuration × training seed × split seed × fold; for example, 6 configurations × 3 training seeds × 5 folds with one split seed is 90 runs.

## Checkpoint and configuration selection

`histopilot/training/fold.py`, `histopilot/candidate_selection.py`, `histopilot/schemas/development.py`

**Checkpoints.** Each fold trains on its fitting groups and scores its validation groups after every epoch, at the target's unit (patient targets aggregate slide predictions first). The recipe's **checkpoint metric** is validation AUROC by default, or validation loss or accuracy. The best checkpoint is the first epoch that reaches the best value. Early stopping ends training after `patience` epochs without improvement (default 8, minimum improvement 0), but not before the minimum epochs (default 1) and never beyond the maximum (default 40). Records saved before these defaults existed keep their original settings (validation loss, 100 epochs, patience 15). Validation AUROC requires every class in each validation fold.

**Small validation folds.** A recipe can set a minimum number of positive validation units for a binary target, with a fixed epoch budget as a fallback. A fold below the minimum trains for exactly that budget without early stopping and keeps its final epoch. A fixed budget can also be set outright. nnMIL recipes can also choose to keep the latest epoch.

**Assessment.** Only after fitting is the assessment fold loaded and predicted by the selected checkpoint. Assessment outcomes never influence checkpoints, stopping or configuration choice.

**Configuration selection.** When a batch has several configurations, the reported one is chosen by validation, never by out-of-fold results. Each configuration's score is the mean, over all of its runs (folds × training seeds × split seeds), of the selection metric at each run's selected checkpoint. The **selection metric** is validation AUROC by default, or validation loss or accuracy. Loss is minimized and the others maximized; ties go to the lower configuration number. The choice is made only once every configuration has complete validation scores. Until then, Results show the lowest-numbered configuration with a warning. A batch that declares a [controlled comparison](#controlled-comparisons) makes no choice: Results report its reference configuration and every other configuration against it.

Validation data drives both checkpointing and configuration choice, and out-of-fold results of the chosen configuration are still development evidence. Their intervals do not account for configuration selection. An independent labeled cohort is needed for an estimate that is free of these choices.

## Model architectures

`histopilot/models/`, `histopilot/models/catalog.py`, `tests/test_mil_parity.py`

Every patch architecture reads masked, padded minibatches and gives a slide the same prediction alone or in a batch. Architectures, losses and inference rules follow the authors' code by default; the few adapted forms are named switches that are off unless chosen. Settings named below are each architecture's own.

- **ABMIL.** Gated attention pooling over projected patch embeddings (Ilse et al., 2018).
- **nnMIL.** Gated attention over sampled feature coordinates that pools the full-dimensional features, with windowed testing (Luo et al., 2025).
- **CLAM-SB, CLAM-MB** (Lu et al., 2021). Gated attention with one branch (SB) or one branch and one-logit slide classifier per class (MB). While fitting, the B most and B least attended patches of each slide train the true class's instance classifier as positive and negative (B = 8); with **Mutually exclusive subtypes**, every other class's classifier also learns that the B most attended patches are negative, and the instance loss is averaged over classes. The loss is w × slide loss + (1 − w) × instance loss, with w = 0.7. The instance loss is the smooth top-1 SVM (margin 1, temperature 1, switching to the hinge for well-separated scores, as the authors' runs used) or cross entropy. As in the reference, the attention branches drop 25% whenever dropout is on; **Attention dropout follows Dropout** uses the recipe's value there. The reference fails on slides with fewer than B patches and labels shared patches both positive and negative below 2B; HistoPilot samples half of such a slide's patches from each end. A CLAM-MB map is its predicted class's branch.
- **TransMIL** (Shao et al., 2021). A class token, the patch tokens in bag order and copies of the first patches that fill a square grid; Transformer layers with Nyström attention (landmark means, iterative Moore-Penrose pseudo-inverse, a depthwise value convolution of width 33); and the pyramid position encoding (depthwise 7×7, 5×5 and 3×3 convolutions) after the first layer. Each slide is its own sequence, so padding never enters attention and the pseudo-inverse's scale depends on that slide only; the reference has no mask and takes that scale over the batch. The Nyström core runs in float32. Layers, heads, landmarks and iterations are editable; the two-layer default is the reference. The attention map, which the reference does not provide, is the class token's exact softmax attention to the patches in the last layer, averaged over heads, with each grid copy's weight added to its source patch; it approximates what the Nyström layer computes.
- **DSMIL** (Li et al., 2021). An instance classifier scores every patch; each class's highest-scoring patch is its critical patch; every patch is attended by its query similarity to that critical patch; a one-dimensional convolution classifies the per-class pooled embeddings. As published, both streams read the patch features directly; **Shared input layer** first applies the projection under Model architecture, as standardized benchmark libraries do. The loss is (1 − w) × slide loss + w × max-instance loss, w = 0.5, with the recipe's loss: one-logit BCE is the reference's binary setting, while several logits use the recipe's loss where the reference applies BCE to each class. Averaged prediction uses the mean of the two streams' probabilities. A map is the attention to the predicted class's critical patch.
- **DTFD-MIL** (Zhang et al., 2022). The input layer is the reference's bias-free Linear and ReLU, with its optional residual blocks. A slide's patches are split at random into pseudo-bags (4 in the authors' code; the paper used 5 and 8); each is pooled with gated attention and classified with the slide's label; each passes its attention-pooled feature (AFS), its highest and lowest scoring patches (MaxMinS) or its highest (MaxS) to a second gated-attention tier that classifies the slide. Patches are ranked by class activation of the first tier's classifier, on the last class as in the reference. Each tier learns from its own loss only, as the paper's objective states and the reference's two optimizers do; HistoPilot steps one optimizer and gives tier 2 the distilled features detached, which yields the same gradients. Testing averages the probabilities of **Test-time pseudo-bag splits** (1, as in the reference); each split comes from a fixed seed instead of a fresh random draw, so predictions repeat. One MaxMinS split moved a slide's probability by up to 0.33 in a synthetic check, so several splits steady clinical predictions. A slide with fewer patches than pseudo-bags gets one pseudo-bag per patch; the reference adds empty ones. A map is each patch's weight in the slide embedding.
- **OceanPath Mamba-2 MIL** (OceanPath's `Mamba2MIL`, after COBRA, Lenz et al., 2025, and Mamba-2, Dao and Gu, 2024). A Linear and LayerNorm patch embedding (384 dimensions); two Mamba-2 blocks with a residual around each; a Linear and LayerNorm; gated attention pooling (256, dropout 0.1); a linear classifier. Each block projects a patch to a gate, a stream and a step; a causal depthwise convolution (width 4) and SiLU give the stream x and the state inputs B and C; the selective scan h_t = exp(Δ_t A) h_(t−1) + Δ_t B_t x_t, y_t = C_t h_t + D x_t runs over 12 heads of 64 channels with a 64-dimensional state, with Δ_t = softplus(step + bias); then RMSNorm of y·SiLU(gate) and an output projection. With **Hilbert ordering** a slide's patches enter the scan sorted along a Hilbert curve of their coordinates (order 16, per-slide min-max scaling), as OceanPath pretrains and fine-tunes; otherwise in feature-file order. The scan is causal and padding comes last, so a slide predicts the same alone or in a batch. **COBRA pooling when predicting** (off, as in OceanPath's recipes) pools the embeddings from before the blocks with the same attention at prediction time. OceanPath runs the scan with mamba-ssm's fused CUDA kernels; HistoPilot computes the same chunked state space duality algorithm with tensor operations, on CPU or GPU. The map is the pooling attention, in the patches' original order.
- **ACMIL** (Zhang et al., 2024). The reference's bias-free input layer, then several gated-attention branches (5) with their own classifiers; while fitting, a share (0.6) of each branch's 10 most attended patches is hidden at random, always leaving one patch visible. The loss adds the branches' mean loss and their mean pairwise attention cosine similarity to the slide loss. The slide classifier pools patches with the branches' mean attention, which is also the map.

The extra loss terms use the recipe's loss with its class weights, focal or BCE form and label smoothing, and every slide's total is weighted by its patient weight. Validation, checkpoint selection and reported probabilities use the slide prediction only.

**Verification.** `tests/test_mil_parity.py` compares every architecture with golden outputs of the authors' code: logits, attention, training losses and gradient norms. `scripts/mil_parity_fixtures.py` recorded them from shallow clones of the official repositories at fixed commits, listed in the fixture. HistoPilot's weights load from the official tensors one to one, and outputs agree within floating-point rounding. That holds in the default configuration and in the switches the reference also has (subtyping, instance losses, gating, distillation modes, residual blocks, value projection).

**OceanPath Mamba-2 MIL** is checked the same way against OceanPath's own model (`tests/test_oceanpath_parity.py`, recorded by `scripts/oceanpath_parity_fixtures.py`): slide embeddings, logits, attention, a training loss and every gradient norm for random weights in several shapes (one and several chunks, padded batches, Hilbert ordering, COBRA pooling, OceanPath's shape), plus one pretrained checkpoint on a synthetic 2,048-patch bag when that file is present. OceanPath's state dicts load into the aggregator strictly, name for name, and a seed draws the same random initialization as OceanPath's `build_classifier`. Outputs agree within 1e-5 of their scale and gradient norms within 1e-4 relative, the float32 rounding of a different summation order; the port in float64 differs from the recorded outputs by as much.

**Pretrained aggregators.** An OceanPath Mamba-2 MIL configuration can start from an OceanPath aggregator state dict, such as OceanPath's self-supervised aggregators (VICReg, JEPA, LeJEPA and multi-crop LeJEPA, pretrained on unlabeled UNI2-h patch features). Review resolves the path inside the configured data roots, with no symbolic links, and records its SHA-256 and size in the frozen batch; launching, every fold and refit worker, and a resume check the file again and refuse a changed or missing one (`MODEL_CHECKPOINT_CHANGED`) rather than train from other weights. Fitting loads the file only when a fold or refit starts fresh; the result records the pin it started from. The classifier starts at random. Weights derived from UNI2-h features are subject to UNI2-h's licence (CC-BY-NC-ND 4.0, non-commercial; Mahmood Lab's terms treat models trained on its outputs as derivatives), so HistoPilot does not ship them: point the recipe to your own copy.

**Comparing initializations.** A controlled comparison whose arms differ only in **Initialization**, **Aggregator checkpoint**, **Aggregator learning rate** or **Freeze aggregator** trains every arm on the same folds and training seeds and reports paired differences against the reference arm, for example random initialization against each pretrained aggregator. These are the model's own settings, so they may differ between arms; the recipe's learning rate, bags and epochs may not. OceanPath's recipes train a random aggregator at the shared learning rate (1e-4) and fine-tune a pretrained one at 1e-5 with its classifier at 1e-4, which the defaults reproduce; set the arms' aggregator rates equal for a comparison that varies the starting weights alone.

**Training protocol.** The architectures are the authors'; their training recipes are not. HistoPilot trains every model with the recipe you set, so models compared in one batch share the optimizer, schedule, epochs and bags. The authors trained on whole bags with batch size 1 and their own optimizers (CLAM: Adam 2e-4, weight decay 1e-5, early stopping on validation loss; TransMIL: Lookahead-RAdam 2e-4; DSMIL: Adam 1e-4, betas 0.5 and 0.9, cosine decay; DTFD-MIL: Adam 1e-4 with a step decay; ACMIL: AdamW 1e-4, cosine decay). HistoPilot can set each of these except Lookahead-RAdam and DTFD-MIL's per-module gradient clipping.

**Reporting these implementations.** A methods paragraph can read: "Multiple-instance models (ABMIL, CLAM, TransMIL, DSMIL, DTFD-MIL, ACMIL and OceanPath's Mamba-2 MIL) were trained with HistoPilot, whose implementations follow the authors' code; given identical weights and inputs, each reproduces the official forward pass, training loss and gradients within floating-point rounding. HistoPilot masks padding so that slides can be batched, handles slides with fewer patches than CLAM's sampled instances or DTFD-MIL's pseudo-bags, and draws DTFD-MIL's test-time pseudo-bag splits from fixed seeds. All models used one pre-specified training protocol [optimizer, learning rate, epochs, training bags] rather than each paper's original recipe." Name any switch you turned on, such as DSMIL's shared input layer. For OceanPath Mamba-2 MIL, also name the pretrained aggregator each arm started from, with its SHA-256, and the aggregator learning rate. OceanPath trained its own supervised baselines on 2,048-patch bags drawn by spatially stratified sampling with batches of 32 slides; HistoPilot's bags sample patches uniformly.

## Out-of-fold predictions

`histopilot/workers/train_batch.py`, `histopilot/scoring.py`

For each configuration, training seed and split seed, the assessment predictions of all folds are pooled into one out-of-fold (OOF) set. It is complete only when every fold finished, and it must cover each assessed slide exactly once with its frozen label; otherwise no OOF metrics are reported. Nothing is filled in for incomplete groups. K-fold, predefined-fold and all-site designs assess every training slide; a held-out assessment, or leave-one-site-out over selected sites, assesses only its held-out slides, and its OOF set covers only those.

**Patient aggregation.** For patient targets, slide predictions are combined per patient by the recipe's rule: `mean_probabilities` (the default, the mean of slide probabilities) or `mean_logits` (the mean of slide log-probabilities, renormalized, which equals a softmax of mean logits). A patient's slides must share one label.

## Cross-validated results

`histopilot/application/experiment_results.py`, `histopilot/cv_summary.py`

The Results tab reports, for the selected configuration of each batch:

| Row | Definition |
| --- | --- |
| Per fold | Metrics each fold recorded on its own assessment fold, with its checkpoint epoch. A leave-one-site-out fold is named by its site; a held-out design has one plan per seed. |
| Per seed | Metrics of one pooled OOF set: one training seed and one split seed, every assessed slide scored once by the fold model that did not train on it |
| Seed average | Mean ± sample SD (n − 1) of the per-seed OOF metrics across all complete training-seed × split-seed replicates, with minimum and maximum |
| Fold average | Mean ± SD of all per-fold metrics. Descriptive only: folds are small, and the pooled OOF value is the more stable estimate. |
| Seed ensemble | Each slide's predictions averaged across seeds (mean probability by default, or mean logit), then scored once. Needs at least two replicates. Its deployable form is the [seed-ensemble predictor](#predictors). |

Headline numbers are seed-average OOF values; AUROC is the primary metric. The Results page also flags patterns without changing any number: a class with mean recall below 0.5, a seed SD of 0.02 or more in AUROC, a fold AUROC range of 0.10 or more, a median checkpoint epoch of 2 or less, a single training seed, and a missing validation-based choice.

## Metrics

`histopilot/cv_summary.py` (`point_metrics`); the training and inference workers validate their predictions in `histopilot/training/module.py` and score them with the same function. All are computed in NumPy.

| Metric | Definition |
| --- | --- |
| AUROC | Binary: the Mann–Whitney AUROC of the positive class, with tied scores counting one half. Multiclass: the unweighted mean of one-vs-rest AUROCs (macro), unavailable if any class lacks positives or negatives. |
| AUPRC | Average precision, Σ ΔTP × precision ÷ positives, with tied scores sharing one threshold (no interpolation). Multiclass: the unweighted mean over classes. |
| Balanced accuracy | Mean recall over the classes present in the labels |
| Macro-F1 | Mean F1 over all target classes. A class that is neither present nor predicted counts as F1 = 0, unlike some libraries' defaults. |
| Accuracy | Fraction of correct decisions |
| Loss | Mean negative log-likelihood of the true class (natural log). Reported, but never resampled or used to rank models. |

Rankings use one-vs-rest log odds computed from saved log-probabilities, which order predictions exactly like the probabilities but keep their order near 0 and 1. **Decisions** use argmax for multiclass targets. For binary targets a unit is positive when P(positive) ≥ the recipe's **decision threshold**, 0.5 by default. The threshold changes decisions and decision-based metrics, never AUROC or AUPRC. Metrics that are undefined for the data, for example when a class is absent, are reported as unavailable with the missing classes listed. Calibration, Brier score and log loss appear in [clinical utility](#clinical-utility) and [recalibration](#recalibration).

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

### Controlled comparisons

`histopilot/application/comparisons.py`, `histopilot/application/experiment_results.py`, `histopilot/cv_summary.py`

A batch can declare a controlled comparison: 2–8 custom configurations (arms), all of them built, one of them the reference, and a primary metric (AUROC by default; AUPRC, balanced accuracy, macro-F1 or accuracy). Every arm trains on the same frozen folds and training seeds.

- **What may differ.** Arms may differ only in the model, the architecture options that model owns (attention width and gating for ABMIL and nnMIL, and each architecture's own settings, such as `nnmil*`, `clam*` or `transmil*`), and the inputs: image, clinical or both, and the clinical fields. When either arm of a pair is clinical only, image settings such as the training bag, dropout and patch augmentation are not compared, because a clinical-only model does not read them. Defaults are filled in before comparing, so an omitted setting equals its default. Any other difference, such as learning rate, batch size, epochs or schedule, blocks the batch at review because it would mix the ablated factor with optimisation choices. Identical arms are blocked too.
- **Reported configuration.** Results report the reference arm, not a validation choice, and summarise every arm with intervals.
- **Differences.** Each other arm is reported as reference − arm on the seed means; a positive value means the reference scored higher. The 95% interval comes from the per-draw differences on shared draws: each bootstrap draw resamples the same units for both arms, as for paired batch comparisons. Fold-paired differences count the folds in which the reference did better, worse or tied.
- **p-value.** A two-sided bootstrap p-value on the primary metric for no difference: twice the smaller of the shares of draws with a difference ≤ 0 and ≥ 0, each share computed as (count + 1) / (draws + 1) so it is never exactly zero, capped at 1. Draws invalid for either arm are excluded. The p-value requires both arms to have scored the same units.
- **Holm adjustment.** The planned contrasts of one batch, one per non-reference arm with a p-value, form one family. Their p-values are sorted ascending, the i-th smallest of m is multiplied by m − i + 1, a running maximum keeps the order, and values are capped at 1.

Like the intervals, the p-values hold the trained models fixed; they do not include the variability of retraining.

### Intervals for scored runs

`histopilot/statistics.py`

Scored runs of patient-unit predictors report patient bootstrap intervals for AUROC and AUPRC. They need verified patient IDs, one prediction per patient and at least two labeled patients per class. A paired comparison of two scored runs on the same patients reuses each draw for both and reports left − right. A sensitivity analysis keeps one slide per patient, chosen by a seeded hash of the patient and slide IDs before any score or outcome is read, and repeats the analysis. Slide-unit runs report no bootstrap intervals.

## Predictors

`histopilot/application/predictors.py`, `application/refits.py`, `training/refit.py`, `training/inference.py`

A predictor is a frozen model built from one complete group: one configuration, training seed and split seed, after all of its folds finished. A seed ensemble pools every group of one configuration. Each batch chooses **Ensemble**, **Refit**, **Both** or **Skip**, and which **configurations to build**: only the configuration chosen by validation (the default) or all of them. A batch of 15 configurations × 3 training seeds with one split seed that chooses Both therefore builds 3 ensembles and 3 refits of its selected configuration, or 45 of each for all configurations. The number of folds does not multiply predictors.

**Ensemble.** The group's fold checkpoints (each fold's selected epoch). Inference averages the members' class probabilities (default) or their log-probabilities, renormalized, per the recipe's ensemble aggregation. For patient targets, patient aggregation follows the ensemble.

**Seed ensemble.** Every seed group of one configuration, pooled into one predictor. Build it from **Experiments → Predictors** once all of the configuration's groups have finished.
- It reuses each group's verified fold checkpoints, so nothing trains. That makes it available whatever the batch's predictor choice, including after the experiment's predictor creation has closed.
- Members average with equal weight under the recipe's ensemble aggregation. Every group holds the same number of folds, so this equals averaging the groups' fold ensembles. It is the deployable form of the seed ensemble that cross-validated results report.
- Each member keeps its own training seed for the bags of a run. That matters only when bags are subsampled.
- A configuration with a single seed group has no seed ensemble.
- A seed ensemble holds at most 256 fold models.

**Refit.** One new model trained on all development slides of the group (the union of every fold's train, validation and assessment slides) with the same recipe and training seed. The testing set is never included. There is no validation data, so the epoch count is fixed in advance:

1. Take each fold's best checkpoint epoch (one-based).
2. Take the chosen percentile of those epochs with linear interpolation: P50 (median), P75, P90, P100 (maximum) or any value from 1 to 100.
3. Round up.

For fold best epochs `[2, 3, 5, 8]`, P50 gives 4 epochs and P75 gives 6. The refit trains exactly that many epochs with no early stopping. Warmup is shortened to fit the budget, a cosine schedule spans it, and a plateau schedule, which needs validation, becomes a constant learning rate. A fold whose best epoch cannot be verified from its receipt or validation history blocks the refit; the stopped epoch is never substituted.

**Frozen scoring rules.** A predictor carries its recipe's decision threshold and patient aggregation. Runs use them and refuse a different threshold or aggregation (`EVALUATION_THRESHOLD_MISMATCH`, `EVALUATION_AGGREGATION_MISMATCH`). HistoPilot never tunes a threshold on a cohort.

## Applying models

`histopilot/application/evaluation_runs.py`, `histopilot/inference_summary.py`

A run applies one predictor to one cohort with whole slide bags, the frozen class order and the frozen scoring rules. Its job predicts the cohort without reading any label.
- **Labeled cohorts.** The control service then scores the predictions against the labels frozen in the cohort (`application/run_evidence.py`, `run_metrics.py`), using the same functions a worker used.
- **Earlier runs.** Runs on labeled cohorts saved before label-blind prediction keep the metrics their worker computed with the labels. Both kinds report the same numbers for the same predictions.
- **Unlabeled slides.** Metrics use labeled units only; unlabeled units still receive predictions. A labeled cohort keeps slides without a label when its target sets missing or unmapped values to **Keep as unlabeled**. Only testing targets offer this, because training needs every label.
- **Unlabeled cohorts.** A run reads no labels and computes no metrics until a reference standard labels its cohort.

A cohort that shares any slide, or any source slide file, with the predictor's development data is refused. So is any cohort with shared patients when the predictor scores patients, because those predictions would be in-sample.

In a patient-grouped design, a slide-level predictor may predict new slides of development patients:
- They are flagged.
- A scored run leaves them out of every metric, subgroup, clinical-utility report and paired comparison.
- The metrics count them (`developmentExcluded`).

Slide-unit designs do not check or flag shared patients, so exclude them yourself when patient independence matters.

A label-blind scored run's downloads come from the service:
- `metrics.json` is the scored document.
- The slide and patient prediction tables add each row's label, its development-patient flag and whether it was scored.

### Reference standards

`histopilot/application/references.py`, `run_evidence.join_reference`

A reference standard is a frozen record that labels one cohort after it was frozen. It maps one column of one or more frozen datasets to a class set, matched to the cohort's slides by slide ID. Missing values, values left unmapped and cohort slides absent from the chosen datasets stay unlabeled; slides listed twice block saving. Patient IDs always come from the cohort's frozen membership, never from the new datasets.

A run can be scored against a reference of its own cohort (same record and content hash) whose classes are the run's, in any order; labels are placed in the run's class order.
- **Same functions.** Scores, subgroups, case review, clinical utility and downloads use the functions that score the cohort's own labels, on the same verified predictions. A reference that repeats the cohort's labels gives exactly the cohort's scores. Nothing is predicted again.
- **Patients.** Under a patient target, a patient whose slides the reference labels differently is left unlabeled (`conflictingPatients`), because no worker aggregation rule applies. Patient records are always rebuilt from the relabeled slides.
- **Development patients.** Units from development patients are left out, as they are for the cohort's labels.
- **Intervals.** A run on an unlabeled cohort recorded no bootstrap settings, so its reference scores use the settings its predictor recorded, or the defaults.
- **Storage.** Scores are kept beside the run under a key of the predictions' SHA-256, the reference's content hash and the analysis settings, so they are computed once.

**Agreement** (`EvaluationRunService.agreement`) pairs the run's decisions (frozen threshold, frozen patient aggregation) with the cohort's labels and every usable reference, over the units both sides of a pair label. Development units are left out.

| Quantity | Definition |
| --- | --- |
| Percent agreement | Matching units ÷ units labeled by both |
| Cohen's κ | (p_o − p_e) ÷ (1 − p_e), where p_e sums the products of both sides' class frequencies |
| Linear weighted κ | Three or more classes only: 1 − Σ w·o ÷ Σ w·e with weights \|i − j\| ÷ (k − 1) over the class order, so near misses count as partial agreement. Meaningful only for ordered classes listed in their order: the order is the target's class list, and classes taken from a field follow its listed values, often alphabetical (high, low, moderate is not a scale). |

The label-free summary of a run (`POST …/inference/summary`) compares it with another run on the same cohort when given `comparisonId`: percent agreement, Cohen's κ and linear weighted κ as above, over the units both runs predict, with the cross-tabulation (rows this run, columns the other). This is how two models applied to one unlabeled cohort compare.

**Performance by subgroup** (`application/run_performance.py`) recomputes the same metrics within each value of one frozen dataset attribute, over exactly the records the run scores. Groups are ordered by their labeled count; the 50 largest are listed and the rest are pooled as Other. Groups with fewer than 10 labeled units are marked as unstable. The breakdown is descriptive: it applies no multiplicity correction, and comparing many groups finds gaps by chance.

**Compare methods** contrasts an ensemble with a refit only within the same cohort and scoring unit, and only for matching groups, using the cohort's own labels; runs scored against reference standards are not compared. Choosing between strategies on a cohort uses up that cohort's independence for the chosen strategy.

Every run, labeled cohort or not, describes its predictions the same way. New slides of development patients are flagged on the run page, in case review (the `developmentPatients` filter) and in its export; the worker's raw prediction files do not carry the flag. The descriptions are:

| Quantity | Definition |
| --- | --- |
| Confidence | The probability of the predicted class |
| Margin | Distance from the decision boundary on a 0–1 scale. Binary: (p − t) ÷ (1 − t) above the threshold t, (t − p) ÷ t below it. Multiclass: top probability minus the runner-up. |
| Member agreement | For ensembles, how many fold members make the same decision, and the spread of their probabilities |
| Threshold sweep | Binary predicted-positive counts at thresholds 0.05 to 0.95 |
| Run agreement | Agreement and Cohen's κ between two runs on the same cohort |
| Label agreement | See [reference standards](#reference-standards): the run and every label source, pair by pair |

None of these is an accuracy estimate.

## Clinical utility

`histopilot/application/clinical.py`

A clinical utility report reads the checksummed predictions of a completed scored run for one unit (slide or patient) and one outcome class. Outcomes come from the cohort's labels or, when the analysis names one, a [reference standard](#reference-standards). For multiclass targets the chosen class is compared with all others. It never refits, recalibrates or tunes a threshold; the operating point is the run's frozen threshold unless you enter a descriptive override, which is labeled as such. Unlabeled units are counted and excluded. Calibration after recalibration is reported separately; see [recalibration](#recalibration).

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

## Recalibration

`histopilot/calibration.py`, `EvaluationRunService.recalibration`

A scored run's Performance view shows how well its risks match outcomes, as predicted and after recalibration. The map is fitted on the predictor's out-of-fold development predictions at the run's scoring unit, never on the cohort it is scored on:

- **Binary targets: Platt scaling**, p′ = σ(a · logit p + b), fitted by maximum likelihood with Platt's targets (N₊ + 1) ÷ (N₊ + 2) and 1 ÷ (N₋ + 2), so the map stays finite when development predictions separate the classes.
- **Multiclass targets: temperature scaling**, p′ = softmax(log p ÷ T), with T in [0.05, 20] minimizing the development log loss.

**Source predictions.** A fold-ensemble or refit predictor uses the OOF predictions of its source configuration, training seed and split seed; a seed ensemble averages the OOF predictions of its seed groups as it averages their models. Each group is verified as for its OOF export. A refit model is not one of the fold models whose predictions fit the map, so its map is an approximation.

**What changes.** Only risks. Decisions, the frozen threshold and ranking metrics keep the original probabilities: a Platt slope a > 0 keeps every binary ranking, and temperature scaling keeps every decision. For the cohort, and in-sample for development, the report gives the Brier score (summed over classes for multiclass targets), log loss, ECE over 10 equal-width bins, observed ÷ expected, and the calibration slope and intercept of outcome ~ logit(risk), which are unavailable when the outcome is separable. For multiclass targets the risk is the predicted class's confidence and the outcome whether it is right. Labels come from the cohort or the chosen reference standard, and development patients are left out, as from every metric. Reports are kept beside the run by a key of its predictions, predictor, labels and unit.

## Dataset shift

`histopilot/shift.py`, `application/shift.py`

A run's dataset shift compares its cohort with the slides its predictor was developed on, in the feature space every MIL model here reads. It explains a drop in performance on a new cohort, or warns that one may come. It tests nothing and corrects nothing.

- **Slides.** The development slides are exactly those in the predictor's training batch plan (the members of every fold, verified by the plan's hash). The cohort slides are the cohort's, as the run reads them, minus any development slide. Each slide is represented by the mean of its patch features, accumulated in float64.
- **Atypicality.** A slide's mean cosine distance to its 10 nearest development slides; a development slide leaves itself out. The threshold is the 99th percentile of the development slides' own distances, so by construction about 1% of development slides lie beyond it. A cohort slide past the threshold lies *beyond the development range*. The share of cohort slides beyond it carries a Wilson 95% interval, and each slide's percentile is the share of development distances at or below its own.
- **Separability (domain AUROC).** Development slides are labelled 0 and cohort slides 1. In each of 5 stratified folds, the training fold is standardized and reduced to its 32 leading principal components (dropping components with no variance), and a logistic regression with an L2 penalty of 1 (intercept unpenalized) is fitted by Newton's method. Held-out slides are scored in that space. The out-of-fold scores give the AUROC, and its 95% interval resamples slides within each set 1,000 times, holding the scores fixed. A value of 0.5 means a linear model cannot tell the two sets apart.
- **Plane.** Both sets projected on the development slides' first two principal components.
- **Verdict.** Descriptive. *Strong* when the domain AUROC is at least 0.85 or at least 25% of cohort slides lie beyond the range; *moderate* when it is at least 0.65 or at least 5%; otherwise *little*.
- **Performance by atypicality.** For a scored run, the service (never the job) joins the per-slide distances with the run's labeled units. A unit's atypicality is the mean percentile of its slides. Units are ordered by it and cut into thirds, and separately split into those with any slide beyond the range and the rest. Each group gets the run's metrics, over exactly the records the run scores. Small groups give unstable estimates.
- **Record and job.** An analysis is a frozen `shift-analysis` record, content-addressed by the training batch (with its plan hash) and by the cohort and feature records (by content hash). Runs that share these share one analysis; names stay outside the record. Its job is a Task Center CPU task that re-verifies every feature file's stamp before and after reading it, reads no label, and writes `shift.json` and `slide-embeddings.npz` with SHA-256 receipts. The service serves the report only through its receipt.

## Model cards

`application/model_cards.py`

A frozen predictor's model card puts on one page what a methods supplement or a model registry asks of a model:

- the architecture and its checkpoints (SHA-256);
- the task;
- development performance (out of fold, with the patient bootstrap intervals of the experiment's results);
- every run of Apply models, with its metrics and intervals, calibration as predicted, and dataset shift;
- an intended-use statement (research use only);
- limitations drawn from that evidence, such as a single training seed or cohorts with moderate or strong shift;
- provenance: content hashes, the training plan hash and runtime versions.

Nothing on it is computed for the card. Each figure is one HistoPilot reports elsewhere, named by its record.
