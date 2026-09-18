# Local workflow guide

Practical instructions for working with your own data in HistoPilot. For installation, manual server startup and remote access, see [deployment and development](deployment.md). To explore first, choose **Open BLCA demo**: its [synthetic walkthrough](BLCA_DEMO.md) requires no data folders, training environment or GPU.

## Workspace and data access

The default workspace is `~/.histopilot/workspace`, with configuration in `~/.histopilot/config.toml`. Start the service yourself in a normal terminal, allowing the research directories you need:

```bash
uv run histopilot serve --workspace /path/to/histopilot-workspace \
  --data-root /path/to/research-data --no-browser
```

Replace example paths with directories on the machine running the Python service. Repeat `--data-root` to allow additional directories. To retain access across restarts, merge this into the existing configuration:

```toml
[storage]
data_roots = ["/path/to/research-data"]
```

Explicit `--data-root` arguments override that configured list. Without source roots, you can create a project or explore the demo, but the data and slide folder pickers have no source locations. Add roots when you next restart the service manually; use `--dev` as well if developing through Vite.

**Create a project** with a name and an exact storage folder. Choose a new folder with an existing parent, or an existing empty folder, within the workspace or a configured data root. The folder's `histopilot-project.json` owns its setup. Optional data, slide and feature folders must be within configured data roots; selecting one records a path reference without uploading or copying its contents.

Folder pickers browse the **service's filesystem**, even when the browser runs on another computer. To create a destination, open its parent, choose **New folder**, enter a name, then choose **Use this folder**. Creation requires write access and never replaces an existing file or folder. Load projects from the recent list or their saved folders; `?project=<id>#overview` preserves project selection on refresh.

Keep the service bound to loopback. Remote work uses SSH or VS Code port forwarding; HistoPilot currently supports a single-user local service. See the [example configuration](../examples/config.toml) and [workspace layout](workspace.md) for details.

## Import metadata and link slide files

In **Datasets**, import a CSV/XLSX table, map slide and patient identifiers, review attributes and reconciliation, and freeze a dataset version. Use verified patient identifiers where available; slide fallback groups do not establish patient independence.

By default, slide linkage scans the selected slide folder and matches `Slide_ID` to a filename without its extension. Duplicate stems in different subfolders are ambiguous. If metadata already identifies exact files, map **Slide file column** instead:

| Slide_ID | Slide_Path |
| --- | --- |
| `BLCA-001` | `development/BLCA-001.svs` |
| `BLCA-002` | `test/BLCA-002.svs` |

Paths are relative to the slide folder; permitted absolute paths can be used without one. Keep `Slide_ID` equal to the filename stem because TRIDENT derives its output names from that stem. Explicit paths avoid scanning unrelated copies and the folder scan's 10,000-file limit. Empty or missing paths, unsupported extensions, paths leaving the selected folder, and duplicate file references appear in the preview. Common column names such as `Slide_Path`, `slidePath` and `wsi` are detected automatically.

## Define development targets and splits

**Targets & splits** selects development records from a frozen dataset, defines labels and the positive class, and freezes patient-grouped assignments. Split seeds control those assignments; training seeds belong to Experiments and never redraw them.

Current development protocols contain fitting, early-stop validation and development assessment roles. They do not reserve a final test cohort. Five split designs are available: k-fold, Monte Carlo, leave-one-site/cohort-out, nested k-fold and development holdout. Native training currently executes development-only k-fold protocols; the other designs remain available for study-design review. Nested-CV execution still needs its configuration-selection dependencies connected. See [development targets and split strategies](split-strategies.md) for exact semantics, fixed validation and legacy protocol compatibility.

When selecting a feature version, **Develop on slides that have these features** makes feature coverage part of the population definition before labels and splits. Without that option, every eligible development slide must have features. Review excluded counts before freezing: packing does not repair missing coverage.

Test cohorts are defined separately in **Test cohorts**, and require a dataset but no development model or features. Choose records, define prediction targets or an unlabeled prediction cohort, then review and freeze. Model, feature and inference compatibility are checked later in **Evaluate models**.

## Name frozen versions

Choose **Name & freeze version** for a dataset or target/split protocol, or **Name & freeze bundle** for features. Supply a required **Version tag** and optional **Commit note**. Tags appear in saved-record selectors and experiment inputs; they are unique within the same record kind and project, ignoring case.

Tags can contain up to 80 characters and notes up to 2,000. Renaming a saved tag changes its display metadata while preserving its scientific ID, memberships, feature bindings and existing references. Clearing both fields removes the label and note. A cohort's target and split protocol share its version tag.

Freezing identical scientific content reuses the existing version. Edit that version's label to rename it; change the underlying data or settings to create a distinct scientific version. Concurrent label edits are detected, and interrupted saves retain their operation identity and requested label.

## Extract features with TRIDENT

**Slide features** can start while metadata curation is still in progress. Choose **Extract with TRIDENT**, select slide inputs and a dedicated output folder, and review the command and runtime. Run segmentation, patch coordinates, feature extraction or the full pipeline. **Advanced Options** includes segmentation, readers, patch geometry, MPP metadata, cache, workers, GPU settings, checkpoints and slide encoders. Options are checked against the installed TRIDENT parser before submission.

### Select slides and source resolution

The slide folder supplies the default source, including subfolders. A slide list replaces that scan; with neither, a selected dataset's linked slides can supply the source. An optional dataset filters that source to its claimed slides. Preview reports source counts, filtered counts, MPP coverage and slides the dataset does not claim.

TRIDENT output names must be unique by slide stem. If canonical slides sit beside repaired or quarantined copies with the same name, narrow the source folder or supply a list selecting one copy of each slide.

Under **Advanced Options → Slide reading & selection**, choose **Custom list of wsis** to use a server-side CSV:

```csv
wsi,mpp
development/BLCA-001.svs,0.5016
test/BLCA-002.svs,0.252
```

The listed paths are relative to the slide folder and must exist beneath it. Extra columns are ignored. If `mpp` is present, every row needs a positive finite value; a partly filled column is rejected. Without it, source slide metadata supplies resolution, optionally using custom MPP keys. Declared MPP belongs to the saved input identity: changing it requires a new output folder.

### Configure the extraction environment

TRIDENT runs in its own environment. Set its interpreter and checkout before manually starting the service:

```bash
export HISTOPILOT_TRIDENT_PYTHON=/path/to/trident-environment/bin/python
export HISTOPILOT_TRIDENT_ROOT=/path/to/TRIDENT
uv run histopilot serve --data-root /path/to/research-data --no-browser
```

The usual `~/miniconda3/envs/trident` environment and a checkout at `.local/TRIDENT` are also discovered automatically. Model dependencies, checkpoint access and gated-model authentication must be available in that environment. Runtime discovery does not import Torch into the control service.

Extraction workers use `histopilot-pfm-<run-id>` tmux sessions. The job panel shows stages, current slide, progress and available timing estimates. **Troubleshooting** provides raw logs and the reconnect command. `<project>/extractions/<run-id>/` retains the command, selected-slide CSV, input evidence, settings, process records and `worker.log`. Successful process exit is followed by artifact validation; missing/corrupt outputs and unfinished locks fail the run. Failed or cancelled outputs remain available for compatible resume.

TRIDENT's native layout is preserved, for example:

```text
output/
  contours/                       contours_geojson/       thumbnails/
  _config_segmentation.json       _logs_segmentation.txt
  20x_256px_0px_overlap/
    patches/<slide>_patches.h5
    features_uni_v1/<slide>.h5
    visualization/
    _config_coords.json           _config_feats_uni_v1.json
```

**Attach existing features** accepts the output root, a geometry folder or a specific encoder directory. Encoders remain separate, with HDF5 attributes and nearby TRIDENT configuration retained where available. Patch coordinates may be embedded or stored in `patches/*_patches.h5`. Slide encoder outputs in `slide_features_<encoder>` are supported for slide-embedding linear and MLP probes; patch-bag models use patch features.

## Validate features and freeze a bundle

Inspect a feature source against a frozen dataset, or select **Every slide in this folder — no dataset** for a reusable slide store. Dataset scope reports matched, missing and orphan slides. Store scope keeps every readable slide and binds to a cohort later through coverage checks. A feature source can serve multiple dataset versions when it covers every selected slide.

The bundle workflow reviews coverage, optional packing, then the version name. Full validation reads tensor contents; the initial header/coverage preview alone does not. Choose one of:

| Choice | Result |
| --- | --- |
| Features only — skip packing | Validate the source in place, without copying tensors. |
| Features + existing pack | Fully verify a compatible existing pack against the source. |
| Features + new pack | Review destination, precision and space; build and verify a new pack. |

A bundle can contain zero, one or several verified packs. Its frozen record pins source and pack identities, paths, precision and validation evidence. Inputs remain in their original folders; a bundle is not an archive. Freeze rechecks freshness, and changed or missing inputs block planning until inspected and verified again.

### Packing and precision

Portable memory-mapped packs currently support **patch features with coordinates**. Slide vectors use native source files and full native-precision validation. New patch packs preserve float16 or float32 source precision by default; explicit float16 conversion is available. Overflow fails instead of clipping or producing infinity.

An existing pack may preserve or reduce source precision. Verification compares every feature value against the source converted to the pack's dtype, plus exact slide IDs, dimensions, patch counts and coordinates. Higher precision than the source is rejected. Matching folder sizes or tensor shapes alone are insufficient. Genuine four-file OceanPath packs can be attached through content verification; HistoPilot packs must retain both their manifest and checksum metadata.

```text
feature-pack/
  features.bin       # Little-endian float16 or float32 patch rows
  coords.bin         # Matching little-endian int32 coordinate pairs
  index.parquet      # Slide IDs, row offsets and patch counts
  meta.json          # Tensor layout
  manifest.json      # HistoPilot identity and source/validation evidence
  checksums.json     # Payload and manifest checksums
```

Packed coordinates must be nonnegative integers representable as int32. Native validation can accept int64 coordinates without that packed limit. Packing preserves tensor contents and recorded metadata, not the original HDF5 containers byte for byte.

**Experiments → Inputs** owns loading policy: **Auto**, original per-slide files, or an explicit bundled pack. Auto uses native files for features-only bundles and a sole precision-preserving pack when present. Multiple packs or precision changes require a deliberate choice. Memory mapping does not preload the whole pack into RAM or GPU memory.

Packing/validation jobs use `histopilot-pack-<run-id>` tmux sessions and save evidence under `<project>/packing/<run-id>/`. They publish only after validation and checksum readback, into a new or empty destination; completed packs are not overwritten. Cancelled or failed jobs can be retried as new jobs.

Use IDs from saved-version details for the CLI:

```bash
uv run histopilot pack-features FEATURE_VERSION_ID --project PROJECT_ID --validate-only
uv run histopilot pack-features FEATURE_VERSION_ID --project PROJECT_ID --preview
uv run histopilot pack-features FEATURE_VERSION_ID --project PROJECT_ID --output /allowed/new-pack
uv run histopilot pack-features FEATURE_VERSION_ID --project PROJECT_ID --existing-pack /allowed/existing-pack
uv run histopilot feature-jobs --project PROJECT_ID
uv run histopilot feature-jobs --project PROJECT_ID --job PACKING_JOB_ID --cancel
uv run histopilot verify-feature-pack /path/to/relocated-pack
```

Add `--dtype float16` when deliberately creating a reduced-precision pack. Submission requires the running local service and tmux; `verify-feature-pack` is standalone. A HistoPilot pack remains independently verifiable after relocation or loss of its source, while a live source binding still requires current source evidence.

## Plan and run development experiments

An **experiment** owns named **batches**. Each batch expands its configurations across training seeds and frozen split plans. A 3-learning-rate × 2-weight-decay grid, repeated over 3 training seeds and 5 folds, produces 6 configurations and 90 runs. Explicit recipe rows preserve parameter pairings instead of forming a grid.

The workspace follows **Planning → Running → Finished**:

1. Create an experiment, or use an existing experiment as a template for an independent editable plan.
2. In **Inputs**, select and check the development protocol, compatible feature bundle and loading policy.
3. In **Batches**, configure models, recipes, training seeds and compute resources. Save each batch and its **Skip**, **Refit**, **Ensemble** or **Both** predictor choice.
4. Choose **Review & submit → Freeze & submit experiment**. Submission locks scientific inputs, recipes and predictor choices.
5. Follow **Runs** for progress, losses, checkpoints and resource samples. **Results** presents completed development results and ready predictors when submitted work reaches its terminal state.

ABMIL, nnMIL, mean-pooling MIL and max-pooling MIL consume patch bags. Linear and MLP probes consume slide embeddings. Clinical-only and combined clinical/image arms also have explicit recipes. Model and representation compatibility is checked before execution; attention interpretation is supported for image-bearing ABMIL and nnMIL predictors.

For clinical comparisons, declare clinical fields in **Targets & splits**, then choose their numeric or categorical types in the batch editor. Verified patient identities and consistent clinical values across each patient's slides are required. Matched clinical-only, image-only and combined arms share the same frozen, feature-covered population. Clinical-only fitting does not read image embeddings; imputation, scaling and category encoding are fitted on training patients only.

### Prepare the training runtime

Keep optional training dependencies separate from the lightweight service:

```bash
uv venv .venv-training --python 3.13
uv pip install --python .venv-training/bin/python -e '.[training]'
```

The checkout discovers `.venv-training/bin/python`. For another environment, set `HISTOPILOT_TRAINING_PYTHON` before manually starting the service. Its runtime panel probes dependencies and CUDA in a separate process.

### ABMIL batch settings

For a baseline, freeze a development-only k-fold protocol and a verified patch-feature bundle. Set dimensions, attention, dropout, optimizer, training bag, epoch budget and seeds. **Sample patches per bag** uses a positive patch limit; **Use whole bag for training** retains all patches and needs more memory for large slides. ABMIL sampling is deterministic by slide, training seed and epoch. Validation and assessment use full bags by default; explicit evaluation caps are frozen with the recipe.

New standard recipes use learning rate `3e-4`, weight decay `1e-4`, 40 maximum epochs and patience 8. Advanced controls include FP32/FP16/BF16, gradient accumulation/clipping, schedulers and warmup, minimum training epochs and minimum validation improvement. Saved recipes retain their original settings. nnMIL adds its own feature-window sampling and checkpoint policy; do not assume ABMIL's full-bag evaluation behavior applies to every architecture.

For patient targets, the default training policy gives patients equal expected weight and the default validation aggregation averages slide probabilities within each patient. Validation selects checkpoints and controls early stopping; development assessment folds are predicted afterward. OOF exports require exact held-out coverage for their configuration and seed group. Choosing configurations from OOF results does not make those results an independent final-test estimate.

Choose CPU or GPU, concurrent runs and runs per GPU. Advanced settings include GPU IDs, CPU threads, loader workers and RAM reservations. With one GPU, **Concurrent runs = 6** and **Runs per GPU = 1** still allow only one GPU run at a time. Configured slots do not establish VRAM capacity. Suggested runtime settings distinguish measurements from estimates and remain editable.

Training batches run in `hp-train-*` tmux sessions, with logs, `best.ckpt`, `last.ckpt`, progress, predictions and OOF outputs under `<project>/training/<batch-id>/`. Resource reservations coordinate HistoPilot training across projects; RAM reservations are scheduling allowances, not operating-system limits. Resource charts show recorded measurements, including gaps; an updated UI cannot recreate telemetry an older worker did not collect.

```bash
uv run histopilot train-batch BATCH_ID --project PROJECT_ID
uv run histopilot training-status BATCH_ID --project PROJECT_ID
uv run histopilot training-status BATCH_ID --project PROJECT_ID --results
uv run histopilot training-status BATCH_ID --project PROJECT_ID --cancel
uv run histopilot train-batch BATCH_ID --project PROJECT_ID --resume
```

These commands operate on an existing frozen batch through the same API as the browser. Add `--url http://127.0.0.1:PORT` for a different service port.

### Predictors and independent evaluation

Each batch's saved policy can publish fold ensembles, refit on all development data, or both. A refit budget is a chosen percentile of the folds' selected checkpoint epochs: best-validation epochs for standard ABMIL, and the frozen checkpoint policy for nnMIL. It is an epoch budget, not a fraction of the data. Predictor identity retains experiment, batch, configuration, training seed, split seed and method.

In **Evaluate models**, select ready predictors and a frozen test cohort, then review target compatibility, feature representation, extracted/packed slide coverage, patient conventions and development overlap. The reviewed predictor list is fixed at submission. Completed results expose slide/patient metrics and prediction exports; evaluation predictions feed **Clinical utility**, and compatible predictors feed **Model interpretation**.

See [experiment lifecycle](EXPERIMENT_LIFECYCLE.md) for submission and tracking, and [experiments, predictors and evaluations](MODEL_DEVELOPMENT.md) for ownership, refit epoch calculations and evaluation batches.

## Recovery and record management

Unsaved inputs in **Datasets**, **Targets & splits** and **Test cohorts** are retained for the browser tab. Returning or reloading opens the module library with a recovery action. Review/freeze steps require fresh server checks; a recovered dataset import rereads its source. Opening another record or creating one saves current input as a project draft first. Browser recovery copies do not replace saved project records. Slide-feature acquisition settings do not yet have the same recovery support.

**Cancel** stops pending work and requests active workers to exit while retaining logs, completed work and available checkpoints. **Resume unfinished runs** preserves completed folds and restores the last completed epoch, replaying an interrupted epoch. Current executions archive their Python worker code and verify original inputs, plans, runtime dependencies and execution location before resuming. If that setup has changed incompatibly, create a new batch. Updating the application does not change already-running worker code.

Long-running extraction, packing and training workers survive browser or service disconnection through tmux. Use the session name and `tmux attach -t SESSION_NAME` shown in the job details to reconnect. Check `tmux ls` before manually launching compute so an existing job is not duplicated. tmux does not survive a workstation reboot; durable logs and checkpoints provide the available recovery boundary. The **HistoPilot server is always started and restarted manually in your terminal, never hosted in tmux**.

Use a record's **Manage** action for archive, recoverable Trash or restore, or **Project tools → Workspace cleanup** to review related records. Dependencies and active jobs are checked again at confirmation. Record cleanup preserves source files, slides, features, packs, checkpoints and logs; it does not reclaim their disk space. Whole-project cleanup changes project visibility while retaining child states. See [workspace cleanup rules](WORKSPACE_CLEANUP.md).
