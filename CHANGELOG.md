# Changelog

All notable changes to HistoPilot are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **OceanPath Mamba-2 MIL.** OceanPath's Mamba-2 slide aggregator as an image model, ported to plain PyTorch (no mamba-ssm or CUDA kernels) and checked against OceanPath's own model by golden outputs: embeddings, attention, losses, gradients and its random initialization. Patches are ordered along a Hilbert curve of their coordinates, as OceanPath does, so coordinates now travel with the features to models that read them.
- **Pretrained aggregators.** An OceanPath Mamba-2 MIL configuration can start from an OceanPath aggregator checkpoint, such as OceanPath's self-supervised aggregators, with its own aggregator learning rate or a frozen aggregator. Review pins the file by SHA-256 in the frozen batch; launches, folds, refits and resumes refuse a changed or missing file (`MODEL_CHECKPOINT_CHANGED`). Controlled comparisons can set random and pretrained initializations against each other on the same folds and seeds.
- **`save_design` agent tool.** Saves an experiment design as a draft, or saves it over the draft it names, writing only what changed, and says whether it wrote anything.

### Changed

- **Agents' requests are checked when they are filed.** A commit from a scoped token with `commit` scope first runs the commit's own checks, writing nothing: freezing or starting an experiment, Apply models runs and Task Center actions that could only fail when a person approves them are refused at once with the commit's error and code, such as `EXPERIMENT_SETUP_REQUIRED` for a start of a design nobody froze, and nothing is parked. Approval still checks everything again, the training runtime included.
- **`preview_design` writes nothing.** The agent tool now previews a saved experiment draft, named by ID or name, and returns the preview hash a person confirms; saving moved to `save_design`. MCP clients that passed a design to `preview_design` pass it to `save_design` instead.
- **Controlled comparisons in results summaries.** `experiment_results` summaries for agents keep each arm against the reference arm (reference − arm, paired interval, p-value, Holm-adjusted p-value and fold agreement), and `histopilot experiment results` prints them.

### Fixed

- **Saving an unchanged experiment draft keeps its revision.** A save through the browser, the CLI or an agent that changes nothing no longer makes a new revision, so preview hashes and pending requests made for the draft stay current.

## [0.1.0] - 2026-10-29

The first release. HistoPilot is a local-first research workspace for computational pathology: it takes a slide table to cross-validated, compared and applied multiple instance learning (MIL) models on foundation-model features, and keeps every step frozen, versioned and reproducible. You drive it from the browser, the terminal or an AI agent, all through one local API.

### Added

- **Service and browser UI.** One local FastAPI service serves the React interface and the `/api/v1` API on a loopback address. Host, Origin and fetch-metadata checks and a per-process session token guard it, and `--login` adds sign-in through a printed link. A project roadmap shows where a study stands and opens each stage.
- **Projects.** A project lives in its own folder, which can be moved and reopened, and keeps its records in a SQLite store with journaled, checksummed publication. Every stage ends by freezing an immutable version with a tag. Study backups export, verify and restore archives; workspace cleanup archives, trashes and restores records; drafts are recovered after an interrupted session.
- **Datasets.** Import CSV or XLSX tables, map slide and patient identifiers, reconcile rows with slide files read in place, and freeze versioned datasets. A slide viewer for OpenSlide formats and SDPC serves visual QC.
- **Slide features.** Extract foundation-model features with TRIDENT as queued GPU tasks, or attach existing patch features or slide embeddings. Validate them, optionally pack them into verified memory-mapped packs, and freeze feature bundles.
- **Targets & splits.** Prediction targets with explicit class mappings and positive classes; slide- or patient-unit training and testing sets from random, metadata-condition or predefined splits. The testing set becomes a cohort for Apply models.
- **Experiments.** Designs with k-fold, predefined-fold, leave-one-site-out or held-out assessment; parameter grids and custom configurations over training and split seeds; ABMIL, nnMIL, CLAM-SB, CLAM-MB, TransMIL, DSMIL, DTFD-MIL, ACMIL, mean- and max-pooling MIL, and linear and MLP probes on slide embeddings, reading the image, clinical variables or both. Results report out-of-fold metrics with seed means and bootstrap intervals, and controlled comparisons add paired intervals and Holm-adjusted p-values. Batches build fold-ensemble, refit and seed-ensemble predictors. Objectives, roles and series organize the experiments of a study.
- **Task Center.** One machine-level queue and runner for all compute. It admits tasks against GPU slots, GPU memory, RAM and CPU threads using measured estimates; holds, cancels, retries and reorders work; keeps each task's log and measured resources; and recovers from reboots, crashes, out-of-memory errors and lost GPUs. Submitted experiments run from archived, fingerprinted code.
- **Apply models.** Labeled and unlabeled cohorts; reviewed batches of predictor runs, whose jobs never read labels; performance, subgroup, agreement, prediction and case-review views; reference standards that attach labels after a cohort is frozen; recalibration fitted on out-of-fold development predictions; and clinical utility reports with calibration, operating points and decision curves.
- **Model interpretation.** Attention maps and top patches of attention-based predictors, per model or pooled across several models, in the slide viewer.
- **Command line.** `histopilot <noun> <verb>` commands for every stage, spec files that can live under version control, a preview before every change, and `--json` output with stable exit codes.
- **AI agents.** Per-project AI-exposure levels with pseudonymized identifiers, scoped tokens, approvals for changes an agent requests, and an audit log. An MCP server (`histopilot agent serve`) and a Claude Code plugin with the `histopilot` skill connect agents.
- **Synthetic BLCA demo.** A read-only walkthrough of a synthetic study that needs no data, training environment or GPU.
- **Containers.** A CPU image for the service and UI, a CUDA image for the Task Center runner and training, and a Compose setup with a GPU worker and a demo profile. Container mode (`HISTOPILOT_CONTAINER=1`) listens on the container's interfaces and keeps the loopback Host and Origin checks.
- **License.** HistoPilot is released under the MIT license.

[Unreleased]: https://github.com/Lyce24/HistoPilot/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Lyce24/HistoPilot/releases/tag/v0.1.0
