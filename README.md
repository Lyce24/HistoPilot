<p align="center">
  <img src="web/public/favicon.svg" width="56" height="56" alt="HistoPilot compass" />
</p>

<h1 align="center">HistoPilot</h1>
<p align="center"><strong>From pathology slides to model evidence.</strong></p>
<p align="center">
  <a href="#get-started">Get started</a> ·
  <a href="#the-workflow">The workflow</a> ·
  <a href="#controlled-comparisons">Controlled comparisons</a> ·
  <a href="#the-task-center">Task Center</a> ·
  <a href="#documentation">Documentation</a>
</p>

HistoPilot is a local research workspace for computational pathology. It takes you from a slide table to evaluated models:
1. Import slide and patient metadata.
2. Extract or attach foundation-model patch features.
3. Fix your training and testing sets.
4. Train multiple instance learning (MIL) models with cross-validation.
5. Compare designs with controlled ablations.
6. Evaluate, apply and interpret the resulting predictors.

It runs as a single-user service on your own workstation. Slides, features and results stay on that machine. Source files are read in place, never uploaded or copied.

![Project roadmap of a synthetic grading study, with every required step complete](docs/assets/app/roadmap.png)

<sub>Every screenshot on this page comes from the current interface running on a fully synthetic study. That study has 205 generated slides across four sites, with no real patients, images or results. Its numbers illustrate the software, not a finding.</sub>

## Get started

**Requirements:**
- Linux or WSL
- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- Node.js 22.12+ with npm
- tmux

Training and feature extraction also need an NVIDIA GPU in practice. The BLCA demo needs no GPU.

```bash
git clone https://github.com/Lyce24/HistoPilot.git
cd HistoPilot
uv sync --locked                    # the service
npm --prefix web ci                 # the browser UI's build tools
UV_PROJECT_ENVIRONMENT=.venv-training uv sync --locked --extra training   # training, evaluation, attention
```

Add `--extra imaging` to the first `uv sync` to view SVS, TIFF and other OpenSlide slides in the browser. Feature extraction uses a separate TRIDENT checkout; see [TRIDENT feature extraction](docs/deployment.md#trident-feature-extraction).

Start the service in a terminal and allow the folders that hold your data:

```bash
bash serve.sh --data-root /path/to/research-data
```

Open `http://127.0.0.1:8787`. Choose **Start a new project** and pick an empty folder. Or choose **Open BLCA demo** for a read-only [synthetic walkthrough](docs/blca-demo.md) that needs no data or GPU. The folder pickers browse the machine running HistoPilot, not the computer running the browser.

`serve.sh` rebuilds the browser UI whenever `web/` has changed, then starts the service in the foreground. Stop it with **Ctrl+C**. After pulling an update:
1. Rerun the setup commands if the dependencies changed.
2. Start `bash serve.sh` again.

The Task Center runner restarts itself on the new code. Running tasks are adopted, not stopped. `bash serve.sh --help` lists the options, such as `--port`, `--workspace` and `--no-build`.

## The workflow

The **Project roadmap** groups the modules into seven steps. Each step ends by **freezing** an immutable version, and freezing never starts compute.

| Step | Module | You produce |
| --- | --- | --- |
| 01 | **Datasets** | A frozen dataset: slide and patient records from a CSV/XLSX table, linked to slide files |
| 02 | **Slide features** · **Targets & splits** | A validated feature bundle (TRIDENT extraction or existing HDF5 features). Separately, a frozen target with fixed training and testing sets. |
| 03 | **Experimental Setup** | A frozen design: inputs, cross-validation folds, model recipes, controlled comparisons and predictor choices |
| 04 | **Experiments** | Trained folds, cross-validated results, and ensemble or refit predictors |
| 05 | **Evaluate models** · **Run inference** | Test-cohort metrics with confidence intervals, or label-free predictions for new slides |
| 06 | **Clinical utility** | Calibration, operating points and decision curves |
| 07 | **Model interpretation** | Attention maps and top patches on the slides |

The training and testing sets are fixed from dataset records alone, before any features or models exist. Splits are made by slide or by patient. By patient, all of a patient's slides stay on one side of every split. The testing set becomes a reserved test cohort. Folds are drawn only from the training set.

<p align="center">
  <img src="docs/assets/app/targets-splits.png" width="820" alt="A frozen patient-level target and split: sites A and B for development, site C as an external test cohort" />
</p>

Models include ABMIL, nnMIL, mean- and max-pooling MIL, and linear and MLP probes on slide embeddings. A recipe can read the image, a set of clinical variables, or both.

## Controlled comparisons

HistoPilot is built to ablate design decisions, not just to train one model. Start from one configuration and tick the models and inputs to compare under **Ablation arms**. HistoPilot then creates one configuration per arm:
- Every other setting is copied from your configuration.
- Every arm trains on the same frozen folds and training seeds.
- Your configuration becomes the reference.

<p align="center">
  <img src="docs/assets/app/ablation-arms.png" width="820" alt="Ablation arms: ABMIL with clinical and image inputs as the reference, plus mean pooling, image-only and clinical-only arms" />
</p>

The batch review blocks a comparison whose arms differ in anything other than the model and its inputs. For example, a learning rate, epoch budget or bag size that differs between arms would mix the ablated factor with an optimisation choice.

Clinical variables come from the frozen dataset. The prediction target, the testing target and any field that defines the split are refused, with the reason shown. Every fold learns its own missing-value filling, scaling and categories.

<p align="center">
  <img src="docs/assets/app/clinical-inputs.png" width="820" alt="Clinical fields offered from the frozen dataset; Site and Grade are refused because they define the split and the target" />
</p>

**Results** reports every arm against the reference. Each difference comes with a paired 95% interval, computed from the same patient resamples for both arms, and a Holm-adjusted p-value for the planned contrasts.

<p align="center">
  <img src="docs/assets/app/controlled-comparison.png" width="820" alt="Controlled comparison results: seed-mean AUROC per arm, and each arm against the ABMIL reference with paired intervals and Holm-adjusted p-values" />
</p>

Every result also breaks down by training seed and test fold, so you can see how far one run can move. The results page shows out-of-fold (OOF) values as the mean ± SD across seeds, with a patient bootstrap interval and a seed ensemble. It also flags seed and fold variation, very early checkpoints and partial results. OOF results guide development. To get an independent estimate for the configuration you choose, score it on the test cohort.

<p align="center">
  <img src="docs/assets/app/folds-and-seeds.png" width="820" alt="AUROC for every test fold and training seed with the checkpoint epoch each fold's validation chose" />
</p>

## Evaluate, apply and examine

**Evaluate models** scores ready predictors on a labeled test cohort. It reports metrics for the target unit and bootstrap confidence intervals. **Case review** ranks cases by confidence, margin or ensemble disagreement. **Clinical utility** adds:
- calibration;
- precision–recall and operating characteristics;
- decision curves.

| Test-cohort evaluation | Clinical utility |
| :---: | :---: |
| [![Test-cohort evaluation with patient-level metrics and bootstrap intervals](docs/assets/app/evaluation.png)](docs/assets/app/evaluation.png) | [![Decision curve and potential clinical impact across decision thresholds](docs/assets/app/clinical-utility.png)](docs/assets/app/clinical-utility.png) |

**Run inference** applies a predictor to unlabeled slides. It reads no labels and computes no metrics. It describes the predictions instead:
- confidence and margin histograms;
- a threshold sweep;
- agreement between fold models;
- breakdowns by any frozen attribute.

It also flags slides from patients who were used in development.

<p align="center">
  <img src="docs/assets/app/inference.png" width="820" alt="Inference on an unlabeled cohort: confidence by predicted class and positive-class probability against the frozen threshold" />
</p>

## The Task Center

All compute runs through one queue per machine, shared by every project: training folds, refits, evaluations, inference, attention maps, feature extraction, packing and study archives.
- **Admission:** a task starts when a GPU slot, GPU memory, RAM and CPU threads are free.
- **Controls:** each experiment keeps its tasks together, and can be held, stopped, reordered, cancelled or retried.
- **Recovery:** cancelled or interrupted work resumes from its last completed epoch.
- **Reproducibility:** follow-up work on a submitted experiment runs from an archived copy of the code it started with.

<p align="center">
  <img src="docs/assets/app/task-center.png" width="820" alt="Task Center with eight fold runs in progress and an experiment's queued tasks" />
</p>

`histopilot runner status` reports the runner from a terminal. The CLI can also pack features, follow and resume training batches, and verify or restore study archives (`histopilot --help`).

## Development

```bash
uv sync --locked --extra training --extra imaging     # a development environment with Torch
uv run pytest -n auto --dist worksteal -m "not slow"  # fast tier, about 3 minutes
uv run pytest -n auto --dist worksteal                # full suite
uv run ruff check .
npm --prefix web test && npm --prefix web run build
```

See [contributing](CONTRIBUTING.md) for the test helpers, and for the rules that keep stored projects, queued tasks and archived code loadable across releases.

## Documentation

| Guide | For |
| --- | --- |
| [User guide](docs/user-guide.md) | Running a study end to end, stage by stage |
| [Methods](docs/methods.md) | How splits, cross-validation, comparisons, metrics, intervals and predictors are defined |
| [Deployment](docs/deployment.md) | Installation, runtime environments, configuration and remote access |
| [Task Center](docs/task-center.md) | How compute is queued, admitted, cancelled and recovered |
| [BLCA demo](docs/blca-demo.md) | The read-only synthetic walkthrough |
| [Architecture](docs/architecture.md) | Runtime, storage, execution model and code map |
| [API](docs/api.md) | The local HTTP API |
| [Contributing](CONTRIBUTING.md) | Development environments, tests and conventions |

<sub>No project license has been selected. Third-party models and backends retain their own licenses.</sub>
