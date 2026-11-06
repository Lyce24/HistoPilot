# Changelog

All notable changes to HistoPilot are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - Unreleased

Real data in public, and the evidence engineers and reviewers ask for.

### Added

- **Public benchmark on TCGA-NSCLC and CPTAC-NSCLC.** Five aggregators on UNI2-h features, cross-validated on 946 TCGA patients and applied to 451 CPTAC patients, with intervals, GPU memory and training time ([benchmarks](docs/benchmarks.md)). `examples/benchmarks/tcga-nsclc` rebuilds it from spec files and recomputes every number independently.
- **Dataset shift.** A run's cohort compared with its predictor's development slides in feature space: domain AUROC, the share of slides beyond the development range, a principal-plane view and performance within thirds of atypicality. It runs as a CPU task that reads no label (`histopilot run shift-analyze`; [methods](docs/methods.md#dataset-shift)).
- **Model cards.** Every frozen predictor has one, as JSON or Markdown, built only from the records that made and tested it (`histopilot predictor card`; [methods](docs/methods.md#model-cards)).
- **Real studies in the public demo.** `histopilot serve --demo --demo-projects DIR` serves packaged projects read-only in the full application beside the synthetic walkthrough ([deployment](docs/deployment.md#packaged-projects)).

### Changed

- **CI runs the ML code.** Every push runs the backend suite twice: with Torch and Lightning on the runner's CPU, so the reference-parity and training tests run, and without them, so the service is proven to install lean. A Playwright test walks the public demo in Chromium, and the wheel, the release and both container images are smoke-tested before they are published.
- **Containers.** Both images are smoke-tested before they are published: the CPU image must serve the public demo and the CUDA image must import its training stack. Every published image is also tagged `sha-<commit>`, a tag that never moves.

### Fixed

- A project restored from an archive into another folder lost its scored runs, attention maps, recalibration, predictor status and, in experiment results, its intervals, paired contrasts and p-values: readers compared each output's recorded absolute path with the current one. They now accept the same file below its own job or batch folder, and still check its size and SHA-256 where a receipt records them.
- In a controlled comparison, an arm whose paired interval is unavailable says why, naming the arm and the reference, instead of a message about two batches.
- After a restart that renumbered disks (as WSL does), every saved feature and slide file looked changed and nothing could be trained, applied or interpreted. A file is now identified by its inode, size and nanosecond times; its device number is no longer compared.

## [0.1.0] - 2026-10-31

The first release. HistoPilot is a local-first research workspace for computational pathology: it takes a slide table to cross-validated, compared and applied multiple instance learning (MIL) models on foundation-model features, and keeps every step frozen, versioned and reproducible. You drive it from the browser, the terminal or an AI agent, all through one local API.

### Added

- **Service and browser UI.** One local FastAPI service serves the React interface and the `/api/v1` API on a loopback address. Host, Origin and fetch-metadata checks and a per-process session token guard it, and `--login` adds sign-in through a printed link. A project roadmap shows where a study stands and opens each stage.
- **Projects.** A project lives in its own folder, which can be moved and reopened, and keeps its records in a SQLite store with journaled, checksummed publication. Every stage ends by freezing an immutable version with a tag. Study backups export, verify and restore archives; workspace cleanup archives, trashes and restores records; drafts are recovered after an interrupted session.
- **Datasets.** Import CSV or XLSX tables, map slide and patient identifiers, reconcile rows with slide files read in place, and freeze versioned datasets. A slide viewer for OpenSlide formats and SDPC serves visual QC.
- **Slide features.** Extract foundation-model features with TRIDENT as queued GPU tasks, or attach existing patch features or slide embeddings. Validate them, optionally pack them into verified memory-mapped packs, and freeze feature bundles.
- **Targets & splits.** Prediction targets with explicit class mappings and positive classes; slide- or patient-unit training and testing sets from random, metadata-condition or predefined splits. The testing set becomes a cohort for Apply models.
- **Experiments.** Designs with k-fold, predefined-fold, leave-one-site-out or held-out assessment; parameter grids and custom configurations over training and split seeds; ABMIL, nnMIL, CLAM-SB, CLAM-MB, TransMIL, DSMIL, DTFD-MIL, ACMIL, OceanPath Mamba-2 MIL, mean- and max-pooling MIL, and linear and MLP probes on slide embeddings, reading the image, clinical variables or both. Results report out-of-fold metrics with seed means and bootstrap intervals, and controlled comparisons add paired intervals and Holm-adjusted p-values, in the browser, in `histopilot experiment results` and in agents' summaries. Saving an unchanged draft keeps its revision, so a preview hash handed out for it stays valid. Batches build fold-ensemble, refit and seed-ensemble predictors. Objectives, roles and series organize the experiments of a study.
- **OceanPath Mamba-2 MIL.** OceanPath's Mamba-2 slide aggregator as an image model, ported to plain PyTorch (no mamba-ssm or CUDA kernels) and checked against OceanPath's own model by golden outputs: embeddings, attention, losses, gradients and its random initialization. Patches are ordered along a Hilbert curve of their coordinates, as OceanPath does, so coordinates travel with the features to models that read them.
- **Pretrained aggregators.** An OceanPath Mamba-2 MIL configuration can start from an OceanPath aggregator checkpoint, such as OceanPath's self-supervised aggregators, with its own aggregator learning rate or a frozen aggregator. Review pins the file by SHA-256 in the frozen batch; launches, folds, refits and resumes refuse a changed or missing file (`MODEL_CHECKPOINT_CHANGED`). Controlled comparisons can set random and pretrained initializations against each other on the same folds and seeds.
- **Task Center.** One machine-level queue and runner for all compute. It admits tasks against GPU slots, GPU memory, RAM and CPU threads using measured estimates; holds, cancels, retries and reorders work; keeps each task's log and measured resources; and recovers from reboots, crashes, out-of-memory errors and lost GPUs. Submitted experiments run from archived, fingerprinted code.
- **Apply models.** Labeled and unlabeled cohorts; reviewed batches of predictor runs, whose jobs never read labels; performance, subgroup, agreement, prediction and case-review views; reference standards that attach labels after a cohort is frozen; recalibration fitted on out-of-fold development predictions; and clinical utility reports with calibration, operating points and decision curves.
- **Model interpretation.** Attention maps and top patches of attention-based predictors, per model or pooled across several models, in the slide viewer.
- **Command line.** `histopilot <noun> <verb>` commands for every stage, spec files that can live under version control, a preview before every change, and `--json` output with stable exit codes.
- **AI agents.** Per-project AI-exposure levels with pseudonymized identifiers, scoped tokens, approvals for changes an agent requests, and an audit log. An MCP server (`histopilot agent serve`) and a Claude Code plugin with the `histopilot` skill connect agents. A request an agent files is checked as the change itself would be before it waits for approval. Agents save designs with `save_design`, preview a saved draft without writing anything (`preview_design`), and read each arm's paired contrast with the reference in results summaries.
- **Synthetic BLCA demo.** A read-only walkthrough of a synthetic study that needs no data, training environment or GPU. `histopilot serve --demo` serves it publicly and refuses every change and every other route, for example as a Hugging Face Space.
- **Observability.** Structured JSON logs tagged with request and task IDs, and a Prometheus endpoint (`/api/v1/metrics`) with queue depth, task durations and failures, runner health and GPU use.
- **Containers.** A CPU image for the service and UI, a CUDA image for the Task Center runner and training, and a Compose setup with a GPU worker and a demo profile. Container mode (`HISTOPILOT_CONTAINER=1`) listens on the container's interfaces and keeps the loopback Host and Origin checks.
- **License.** HistoPilot is released under the MIT license.

[Unreleased]: https://github.com/Lyce24/HistoPilot/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Lyce24/HistoPilot/releases/tag/v0.1.0
