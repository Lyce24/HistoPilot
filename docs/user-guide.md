# User guide

This guide follows a study from a slide table to evaluated, interpreted models, one stage at a time. It assumes HistoPilot is installed and running; see [deployment](deployment.md). To look around first, choose **Open BLCA demo** on the start page: the [synthetic walkthrough](blca-demo.md) needs no data, training environment or GPU. For the exact definitions behind every number, see [methods](methods.md).

## Before you start

**Allow your data folders.** HistoPilot can only browse folders you allow when you start it:

```bash
bash serve.sh --data-root /path/to/research-data
```

Repeat `--data-root` for more folders, or list them under `[storage] data_roots` in `~/.histopilot/config.toml` ([configuration](deployment.md#configuration)). Folder pickers browse the machine running the service, even when your browser runs on another computer. Nothing is uploaded or copied: HistoPilot records the paths of your tables, slides and features and reads them in place.

**Create a project.** On the start page choose **Start a new project**, give it a name and pick a new or empty folder. To create one, open its parent, choose **New folder**, then **Use this folder**. The project keeps all of its records in that folder. Reopen it later from the recent list or from its folder with **Load an existing project**. Links of the form `?project=<id>#overview` keep your place on refresh.

**Set up compute.** Training and evaluation need the training environment, and extraction needs TRIDENT; see [runtime environments](deployment.md#runtime-environments). The rest of the workflow works without them.

## The workflow at a glance

The **Project roadmap** groups the modules into seven steps. Every module opens a library of its records; open or create one and work through its steps with **Back** and **Next**. A module whose inputs are missing tells you what it needs.

| Step | Modules | You produce |
| --- | --- | --- |
| 01 | Datasets | A frozen dataset: slide and patient records linked to slide files |
| 02 | Slide features and Targets & splits, in either order | A verified feature bundle; a frozen target with fixed training and testing sets |
| 03 | Experimental Setup | A frozen design: inputs, folds, model recipes and predictor choices |
| 04 | Experiments | Trained folds, cross-validated results and predictors |
| 05 | Evaluate models and Run inference | Test-cohort metrics, or predictions for unlabeled slides |
| 06 | Clinical utility (optional) | Calibration, operating-point and net-benefit report |
| 07 | Model interpretation (optional) | Attention maps and top patches |

Under **Project tools** you also find the **Task Center** (all running and queued compute), **Study backups & sources**, **Workspace cleanup** and **System & storage**.

Every stage ends by **freezing** an immutable version. Frozen versions never change; to change something, create a new version or copy the record. Freezing does not start any compute.

## 1. Datasets

In **Datasets**, create a dataset in three steps:

1. **Choose files.** Choose a CSV or XLSX table (up to 20,000 rows and 128 columns; for XLSX, the sheet) and the slide folder. By default HistoPilot scans the folder, including subfolders, and matches each `Slide_ID` to a file name without its extension. Supported slide types are `.svs`, `.tif`, `.tiff`, `.ndpi`, `.mrxs`, `.scn`, `.vms`, `.vmu`, `.bif` and `.sdpc`.
2. **Map columns.** Map the slide ID and patient ID columns; `Slide_ID` and `Patient_ID` are detected automatically. Review each attribute's type and missing-value tokens.
3. **Review & freeze.** Check the reconciliation of table rows and slide files, then **Name & freeze dataset**.

If the table already names exact files, map a **Slide file column** (detected when named `Slide_Path`, `slidePath` or `wsi`):

| Slide_ID | Slide_Path |
| --- | --- |
| `CASE-001-A` | `development/CASE-001-A.svs` |
| `CASE-002-A` | `test/CASE-002-A.svs` |

Paths are relative to the slide folder. Absolute paths also work if they lie inside a data root. A column avoids scanning unrelated copies and the scan limit of 10,000 files. Keep `Slide_ID` equal to the file name without its extension, because TRIDENT names its outputs after the file. Two files with the same name in different subfolders are ambiguous: a slide in the table that matches both blocks freezing, so point the column at the right copy.

**Patients.** Use real patient identifiers wherever you have them. If some slides have no patient ID, HistoPilot asks **Patient ID unresolved** and offers **Continue with Slide ID**. That treats each such slide as its own patient and is recorded as a fallback. Fallback groups are always reported separately, because they cannot show that two sets are patient-disjoint. Patient-level analyses refuse them.

Once frozen, a dataset opens its records, distributions and the **Visual QC & morphology** panel, where you can inspect the linked slides ([slide viewer](#slide-viewer)).

### Version tags

Freezing a dataset, a target/split version or a feature bundle asks for a **Version tag** (required, up to 80 characters) and an optional **Commit note** (up to 2,000). Tags appear in every picker and must be unique per record kind within a project, ignoring case. You can rename a frozen version later; its content, ID and references do not change. Freezing content identical to an existing version reuses that version, provided the tag and note match or the existing version is unlabeled; otherwise HistoPilot asks you to rename the existing version instead.

## 2. Slide features

**Slide features** attaches or extracts foundation-model features and checks them. It can run while you are still preparing targets.

### Extract with TRIDENT

Choose **Extract with a PFM**, then:

1. **Slides.** The slide folder is the default source, including subfolders. A slide list CSV (`custom_list_of_wsis`) on the server replaces the scan:

   ```csv
   wsi,mpp
   development/CASE-001-A.svs,0.25
   test/CASE-002-A.svs,0.25
   ```

   Paths are relative to the slide folder. `mpp` is optional; if present, every row needs a positive value. Without it, resolution comes from the slide metadata. An optional dataset filters the source to its slides.
2. **Stage.** Full pipeline, Segment tissue, Patch coordinates, or Extract features.
3. **Output.** The TRIDENT output folder defaults to `trident/` in the project and must be empty or already belong to this project. Output names are unique by file name, so narrow the source if two copies of a slide share a name.
4. **Advanced Options** cover segmentation, readers, patch geometry, MPP keys, caching, workers, GPUs, checkpoints and slide encoders. Options are checked against the installed TRIDENT version.
5. **Preview** shows the command, the slide counts, MPP coverage, the disk space estimate and any blockers, such as missing slides, an occupied output folder, a missing slide reader or too little disk space.

**Throughput.** `batch_size` (64) is shared by segmentation and feature extraction unless you set `seg_batch_size` or `feat_batch_size`; these count tiles per model call. Automatic `max_workers` uses up to 8 loaders in total, bounded by the CPUs. Keep patch geometry, encoder, MPP and segmentation settings fixed when comparing speed.

Extraction runs as a [Task Center](task-center.md) GPU task and takes only the GPU memory it needs, so it can share a GPU with training. A separate CPU task then validates the outputs. The run shows one status line that links to its task, where you find the log, command and measured resources. `extractions/<run-id>/` in the project keeps the run record: command, slide list, settings, progress and log. **Resume** on a failed, cancelled or interrupted run restarts TRIDENT on the same output. Finished slides are skipped, and partial outputs left by a dead process are renamed `<name>.stale-<epoch>` and redone; nothing is deleted.

TRIDENT keeps its native layout:

```text
<output>/
  contours/  contours_geojson/  thumbnails/
  20x_256px_0px_overlap/
    patches/<slide>_patches.h5
    features_<encoder>/<slide>.h5
```

### Use existing features

Choose **Use existing features** and point to a `features_<encoder>` folder, the geometry folder above it, or the TRIDENT job root. Patch coordinates may be embedded or come from `patches/*_patches.h5`. Choose **One embedding per slide** for `slide_features_<encoder>` outputs, which feed the slide-embedding probes; patch-bag models use patch features.

### Validate and freeze a bundle

The folder's files are the initial selection; a slide list narrows them, and an optional dataset filter narrows them further. With a dataset, coverage reports matched, missing and extra slides. Without one, the bundle is a reusable store, checked against each cohort when it is used. Then pick how the bundle stores tensors:

| Choice | Result |
| --- | --- |
| Features only — skip packing | Validate every tensor in place; nothing is copied |
| Features + existing pack | Verify a compatible existing pack against the source |
| Features + new pack | Build and verify a new memory-mapped pack |

Packs hold patch features with integer coordinates. A new pack keeps the source precision (float16 or float32) unless you deliberately choose float16. Verification compares every value, slide ID, patch count and coordinate against the source; matching sizes alone are never enough. Packing runs as a CPU task, writes to `feature-packs/` in the project by default, and never overwrites a finished pack.

A frozen bundle pins its sources, packs, precision and validation evidence. Inputs stay where they are. If a source changes or disappears, planning is blocked until it is validated again.

## 3. Targets & splits

**Targets & splits** fixes the prediction target and the training and testing sets. It needs no features, folds or hyperparameters. It has four steps:

1. **Dataset & cohort.** Choose the dataset, the **Split unit** and eligibility conditions. The cohort summary updates as you go.
2. **Training & Testing split.** Configure training first, then testing. Each filter shows its live count and share of the cohort.
3. **Prediction Targets.** Map raw values to classes for training. A binary target needs an explicit positive class; HistoPilot never guesses it from value order or frequency.
4. **Review & Freeze.** Check final membership, label exclusions and distributions, then freeze.

**Split unit.** **Slide** (the default) assigns each slide on its own. **Patient** keeps all of a patient's slides in the same set, and scores patients. Slide splits do not keep a patient's slides together; choose **Patient** when patient independence between training and testing matters. See [split units and grouping](methods.md#split-units-and-grouping).

**Methods.**

- **Random split:** a testing percentage and seed, optionally stratified by a metadata field. Use 0% testing if a separate cohort will be the test set.
- **Metadata conditions:** conditions select testing, and training takes the rest or its own conditions. **Use all eligible … outside the training set** makes testing take everything training does not.
- **Predefined partition values:** assign each value of a column to Training, Testing or Exclude.

A slide or patient matching both sets, or one physical file in both, blocks freezing. Training must contain every class. Small strata can make the actual testing share differ from the requested percentage, so check the counts before freezing.

**Testing target.** Choose **Same field and mapping as training**, **Separate testing field and mapping** (it must keep the classes and positive class), or **None · Pure inference**, which keeps the testing slides without reading their labels.

Freezing also creates the testing set's **test cohort**: an evaluation cohort when testing has a target, an inference cohort when it does not. If that step fails, the version stays frozen and offers **Retry test cohort** (or **Create test cohort**). The frozen version never changes when you later add features or design folds.

## 4. Experimental Setup

**Experimental Setup** combines a dataset, a target/split version and a feature bundle into a frozen design. Create a setup or copy an existing one, then work through its steps.

**Inputs & training design.** Choose the three inputs and the training design:

- **K-fold** cross-validation, 2–10 folds (default 5). Other strategies are not available for new setups.
- **Early-stop validation**, the share of each fold's fitting groups held out for early stopping (default 15%).
- **Split seeds** (default 42, up to 10), each giving a different fold assignment.
- **Keep all slides of a case in the same fold**, offered for slide-unit targets.
- **Loading:** Auto, Original feature files, or Packed mmap with a chosen pack.

**Check & continue to hyperparameters** checks that every training slide has features and that the inputs are compatible. Missing features block the setup; they never silently shrink the training set. Folds are drawn from the training set only. See [cross-validation](methods.md#cross-validation).

**Hyperparameters.** Add one or more **batches**. Each batch is a single configuration, a **Parameter grid** (learning rate × weight decay × maximum epochs) or **Custom configurations** (explicit rows that keep their pairings), repeated over **training seeds** (default 42). For example, a 3 × 2 grid over 3 training seeds and 5 folds is 6 configurations and 90 runs.

| Model | Input | Attention maps |
| --- | --- | --- |
| ABMIL, nnMIL | Patch bags | Yes |
| Mean pooling MIL, Max pooling MIL | Patch bags | No |
| Slide-embedding linear probe, MLP probe | One embedding per slide | No |

A standard recipe uses AdamW, learning rate 3e-4, weight decay 1e-4, at most 40 epochs, early stopping with patience 8, the checkpoint with the highest validation AUROC, FP32, 4,096 sampled patches per training bag, equal class weights, averaged slide probabilities for patients and a 0.5 decision threshold. Templates include an ABMIL baseline, nnMIL, a quick check and a learning-rate comparison. Advanced controls include precision, schedulers and warmup, gradient accumulation and clipping, class weighting, sampling, and the small-validation fallback. Validation and assessment use whole bags by default. nnMIL has its own feature-window sampling and checkpoint policy.

The batch editor also lists clinical-only and clinical + image recipes. They need clinical fields declared as model inputs, which new target/split versions cannot yet do, so these recipes cannot currently be run.

Batches carry no compute settings. The Task Center decides at run time how many runs share a GPU, and which device, threads and loaders each gets. Batches saved earlier with compute settings show them as legacy; the Task Center ignores them.

**Predictor choices.** Each batch chooses what to build once its folds finish:

| Choice | Output per configuration × training seed × split seed |
| --- | --- |
| Cross-validation only | No predictor; results only |
| Ensemble (default) | One predictor that averages the fold models |
| Refit | One model retrained on all training slides for a fixed epoch budget |
| Ensemble + refit | Both |

**Configurations to build** decides whether predictors are built only for the configuration that validation selects (the default) or for every configuration in the batch.

A refit budget is a percentile of the folds' best epochs: P50 (preselected), P75, P90, P100 or a custom percentile. It counts epochs, not a share of the data. See [predictors](methods.md#predictors).

**Review & freeze.** Freezing pins inputs, folds, recipes and predictor choices. It starts nothing. To change scientific settings, copy the setup.

## 5. Experiments

**Experiments** lists every experiment. Search by name, ID, notes or tag; filter by state (Active, Archived, Trash, All records) and status; sort by update time, creation time or name. Select two to four experiments and **Compare** their saved inputs and training settings against a baseline.

Open a frozen setup and choose **Start experiment**. HistoPilot rechecks the inputs, features and training runtime, records the code and environment, and queues every fold in the [Task Center](task-center.md). Queue order, hold, cancel, logs and resources live there.

| Status | Meaning |
| --- | --- |
| Ready to run | Frozen, not yet started |
| Queued | Waiting in the Task Center; the reason says for what |
| Running | At least one task is running |
| Held | Held in the Task Center |
| Waiting | Only blocked work remains, the runner will requeue it (for example a busy project), or the runner is stopped |
| Needs attention | A task failed or was interrupted. **Resume** retries it. |
| Cancelled | Its work was cancelled. It can be resumed. |
| Completed | All of its tasks succeeded |

An experiment has three tabs:

- **Runs:** a status line linking to the Task Center, the run table, and each run's training and validation curves. Curves show the latest 2,000 epochs, numbered from 1.
- **Results:** cross-validated results for each batch's validation-selected configuration. Pick a metric (AUROC, AUPRC, balanced accuracy, macro-F1, accuracy) to see the OOF mean ± SD across seeds with a 95% bootstrap interval, every fold and seed, a seed ensemble, per-class results, the OOF confusion matrix, and paired differences between batches. **Things to know** flags partial results, weak classes, seed or fold variation, very early checkpoints and a missing validation choice. Download the fold and seed table (CSV), the full summary (JSON) and per-seed OOF predictions. See [cross-validated results](methods.md#cross-validated-results).
- **Predictors:** the ready ensemble and refit predictors. Predictors are built after their whole batch finishes; each refit runs as its own task. A finished ensemble can be evaluated while refits are still running.

OOF results are development evidence. Choosing a configuration or batch from them does not give an independent estimate; score the chosen predictor on a test cohort for that.

## 6. Test cohorts

Each frozen target/split version already provides its testing set as a cohort. To prepare another one, such as an external cohort, open **Additional test cohorts** from Evaluate models (or **Create inference cohort** from Run inference). The steps are **Test Data** (datasets and conditions), **Prediction Targets** or **Cohort type**, and **Review and Freeze**. An **evaluation cohort** has a target compatible with the predictors; an **inference cohort** has none. Cohorts do not depend on models or features; compatibility is checked when you evaluate.

## 7. Evaluate models

**Evaluate models** scores ready predictors on a labeled cohort:

1. **Experiments:** choose one or more experiments. Each shows its ready ensemble and refit counts.
2. **Evaluation inputs:** choose the methods (both, ensemble or refit), the test cohort and the feature settings. **Advanced: choose individual predictors** picks predictors one by one.
3. **Review and run:** the review lists every predictor and every blocker. It checks target and class encoding, development overlap, encoder, dimension and dtype, feature and pack coverage of the cohort, the frozen threshold and patient aggregation, and the checkpoint hashes. Acknowledge and choose **Run reviewed predictors**. The reviewed list is fixed; predictors created afterwards are not added.
4. **Batch results.**

Each evaluation reports slide and patient predictions, metrics and confusion matrices linked to case review, with checksummed CSV and JSON downloads. Patient-unit evaluations add 95% patient bootstrap intervals for AUROC and AUPRC and a one-slide-per-patient sensitivity analysis. Predictions use the predictor's frozen decision threshold. Where a batch has both an ensemble and a refit of the same group, results compare them within the same cohort and unit. A comparison used to pick a strategy spends that cohort's independence; see [evaluation and inference](methods.md#evaluation-and-inference).

Cancel an evaluation batch from the Task Center. Completed results stay. Retry failed evaluations from their own controls.

## 8. Run inference

Use **Run inference** for slides without labels, when you want the model's predictions rather than a performance estimate.

1. Prepare an inference cohort: freeze a target/split with **None · Pure inference**, or choose **Create inference cohort**.
2. In **Run inference**, choose experiments and methods, then the cohort. The review checks features, packs and development overlap as for evaluation.
3. Run it. Each predictor gets its own run with the same predictions an evaluation of the same inputs would make.

A development slide or source file is never predicted; use that predictor's OOF predictions for it instead. In a patient-grouped design, development patients are refused for patient-level predictors, and allowed but flagged for slide-level ones. Slide-unit designs do not check patients.

A run shows the predicted-class distribution, confidence and margin histograms, for binary targets the positive probability against the frozen threshold with a threshold sweep, fold-member agreement for ensembles, new versus development patients, a breakdown by any frozen attribute, and agreement with Cohen's κ against another run on the same cohort. None of these is an accuracy estimate.

The review queue starts with the predictions **closest to a decision boundary**, measured from the frozen threshold. It can also sort by least or most confident or by fold-member disagreement, and filter to margins below a cutoff (0.2 by default). **Compute attention** queues attention maps for up to 32 listed slides without leaving the run. **Download predictions with metadata (CSV)** gives one row per slide or patient with probabilities, confidence, margin, agreement, the development-patient flag and chosen attributes.

## 9. Clinical utility

**Clinical utility** looks beyond discrimination at a completed evaluation that has no development overlap. Choose the evaluation, the unit (slide or patient) and the positive class, review, and save. HistoPilot reads the evaluation's checksummed predictions; it never refits, recalibrates or tunes a threshold.

The report covers the Brier score and skill score, log loss, observed-to-expected ratio, ROC and precision-recall curves, calibration bins (10 by default) with expected and maximum calibration error, the operating point at the frozen threshold (sensitivity, specificity, predictive values, likelihood ratios and more), operating curves, decision curves with net benefit against treat-all and treat-none, net interventions avoided per 100, and a clinical impact curve. Formulas are in [clinical utility](methods.md#clinical-utility).

An **Operating threshold** you enter is labeled as a descriptive override. Choose threshold ranges that reflect the clinical decision and the relative harms of false positives and false negatives. Wilson intervals appear only when units are independent patients. The focused decision-curve view clips values below −0.10; switch to the full range or the table to see them. Save separate analyses to compare choices; JSON and CSV exports keep the source lineage.

## 10. Model interpretation

**Model interpretation** shows where a predictor's attention falls on the slides. It needs no evaluation or clinical results.

1. **Model & features.** Choose an ABMIL or nnMIL ensemble or refit predictor and a compatible feature bundle, with original files or a verified pack. The slide folder comes from the bundle's dataset.
2. **Slides.** Browse the dataset's slide folder with thumbnails and search, and select up to 128 slides. Selecting starts nothing.
3. **Attention review.** HistoPilot reuses finished or running attention and queues the rest as Task Center tasks. Each slide shows its own status; failed slides can be retried without losing the others. Open a slide for the focused viewer.

Features must match the predictor's encoder, dimension and dtype, and coordinates must be integer level-0 patch origins in the same order as the features. Mismatches block the job; there is no approximate matching. A slide can have up to 2 million patches and 8 GiB of features.

Attention here is **class-independent pooling attention**, normalized within each slide. It shows how the model weighted patches, not tumor probability, class-specific evidence or a causal explanation. An ensemble averages its members' normalized attention, and each member's map stays available. The viewer draws up to 100,000 patches per view and labels partial overlays; zoom in to see every patch. **Top 10** or **Top 20** ranks patches across the whole slide, with numbered boxes and original-colour crops read from the slide itself. Full-map downloads keep every patch.

## Slide viewer

The QC and attention viewers share their controls:

| Action | Control |
| --- | --- |
| Zoom | Mouse wheel or trackpad pinch over the slide; zoom follows the pointer. **+** and **−** also work. |
| Pan | Drag, or the arrow keys |
| Fit | **Fit slide** or **Home**; in the attention viewer **Home** and **Fit patch coverage** frame the extracted patches |
| Review region (QC viewer) | **Draw review region**, drag a rectangle; Escape cancels. **Zoom to selection** focuses it. |

Selecting a slide shows **Preparing slide** while nearby zoom levels are cached, then **Slide ready**. The overview stays usable meanwhile, and finer detail keeps loading as you move. If detail fails to load, **Retry slide detail** tries again. The zoom factor is relative to the fitted slide, not microscope magnification. Slides, patch coverage, attention and review regions share one level-0 coordinate system. See [slide viewing](deployment.md#slide-viewing) for limits.

## Task Center

All extraction, packing, training, refit, evaluation, inference, interpretation and archive work runs in the **Task Center**, one queue shared by every project on the machine. Stage pages show one status chip that links to the relevant tasks. The Task Center holds queue order, hold, cancel, retry, logs, measured resources and history, and a single **Parallel GPU tasks** setting with a measured suggestion. Tasks survive closing the browser, restarting the service and rebooting; interrupted work resumes from its checkpoints. See [Task Center](task-center.md).

## Study backups & sources

**Study backups & sources** exports, verifies and restores project archives, and relinks moved source folders.

- **Export project** writes a verified archive outside the project folder: an integrity-checked copy of the project database and every project file with its checksum. External slides and features are listed as references, not copied, so back them up separately. Export waits until the project has no active jobs.
- **Verify archive** rechecks every checksum.
- **Restore archive** restores into a new folder, never over an existing one, and keeps the project's identity.
- **Source health** lists registered source folders and any unavailable paths. **Relink** points a registration at a moved folder; frozen versions keep the paths they recorded.

Archive jobs run as Task Center tasks. Without the web service, `histopilot verify-study <archive>` and `histopilot restore-study <archive> <new-folder>` queue the same task, even when the original project is gone, and start the runner if it is not running. Follow them with `histopilot runner status` or on the Task Center page.

## Workspace cleanup

Use a record's **Manage** action, or **Project tools → Workspace cleanup**, to organize a project without losing evidence:

| Action | Effect |
| --- | --- |
| Cancel job | Stops a running extraction, packing, training, refit or evaluation job. Completed work, logs and checkpoints are kept. |
| Archive | Hides a record from ordinary lists and pickers. Saved references keep working. Restore at any time. |
| Move to Trash | Hides a record and blocks new use. Restore at any time. |

Nothing is ever purged and no disk space is reclaimed: tables, slides, features, packs, checkpoints and logs stay on disk. The review lists every record that depends on your selection. A record that others still need can only go to Trash together with them (**Include required records and review again**); restoring a record may require restoring its inputs. Active jobs must stop first, checked by process identity rather than by a status file. Dependencies are checked again at confirmation, and a lost confirmation can be retried safely. A whole project can be archived or trashed from the start page; its records keep their own states. See [architecture](architecture.md#persistence) for where lifecycle state is stored.

## Drafts and recovery

- Unsaved input in **Datasets**, **Targets & splits** and **Test cohorts** is kept for the browser tab. Returning to the module offers **Return to current import** (or draft, or test cohort). Review and freeze steps always recheck with the server.
- Setup inputs, batch edits and experiment forms restore automatically with a notice.
- **Slide features** settings are not recovered; finish or save them before leaving the page.

**Cancel** keeps completed work and checkpoints. **Resume** continues from the last completed epoch and replays an interrupted one. Work always resumes with its original code and environment; if the environment changed, restore it or copy the experiment.

## Command line

The CLI talks to the running service through the same API as the browser. Add `--url http://127.0.0.1:PORT` for a non-default port. IDs come from the saved-record details.

```bash
# Feature validation and packing
uv run histopilot pack-features FEATURE_ID --project PROJECT_ID --validate-only
uv run histopilot pack-features FEATURE_ID --project PROJECT_ID --preview
uv run histopilot pack-features FEATURE_ID --project PROJECT_ID --output /path/to/new-pack   # add --dtype float16 to reduce precision
uv run histopilot pack-features FEATURE_ID --project PROJECT_ID --existing-pack /path/to/pack
uv run histopilot feature-jobs --project PROJECT_ID [--job JOB_ID --cancel]
uv run histopilot verify-feature-pack /path/to/pack                                          # standalone

# Development batches
uv run histopilot training-status BATCH_ID --project PROJECT_ID [--results | --cancel]
uv run histopilot train-batch BATCH_ID --project PROJECT_ID --resume
```

An experiment's batches are started by **Start experiment**; use `train-batch --resume` to continue one. `histopilot runner status` reports the Task Center runner ([the runner](task-center.md#the-runner)).
