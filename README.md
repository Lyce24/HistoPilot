<p align="center">
  <img src="web/public/favicon.svg" width="56" height="56" alt="HistoPilot compass" />
</p>

<h1 align="center">HistoPilot</h1>
<p align="center"><strong>From pathology slides to model evidence.</strong></p>
<p align="center">
  <a href="#get-started">Get started</a> ·
  <a href="#the-workflow">The workflow</a> ·
  <a href="docs/blca-demo.md">BLCA walkthrough</a> ·
  <a href="#documentation">Documentation</a>
</p>

HistoPilot is a local research workspace for computational pathology. It takes you from a slide table to evaluated models: import slide and patient metadata, attach or extract foundation-model patch features, train multiple instance learning (MIL) models with cross-validation, then evaluate, apply and interpret the resulting predictors. It runs as a single-user service on your own workstation. Slides, features and results stay on that machine, and source files are referenced in place, never uploaded or copied.

![HistoPilot workspace showing the synthetic BLCA demo](docs/assets/blca/overview.png)

## Get started

**Requirements:** Linux or WSL, Python 3.11+, [uv](https://docs.astral.sh/uv/), Node.js 22.12+ with npm, and tmux. Feature extraction and training also need their own Python environments and, in practice, an NVIDIA GPU; the demo needs neither.

One-time setup:

```bash
git clone https://github.com/Lyce24/HistoPilot.git
cd HistoPilot
uv sync --locked
npm --prefix web ci
```

Start HistoPilot in your terminal:

```bash
bash serve.sh
```

Open `http://127.0.0.1:8787` and choose **Open BLCA demo**. The demo is synthetic and read-only; it needs no research data, model weights or GPU.

Stop the service with **Ctrl+C** and start it again with `bash serve.sh`. Each start rebuilds the browser UI when `web/` has changed. After an update that changes dependencies, repeat the two setup commands. Run `bash serve.sh --help` for options.

### Your own project

Allow the folders that hold your metadata, slides and features:

```bash
bash serve.sh --data-root /path/to/research-data
```

Choose **Start a new project** and pick an empty or new project folder. The folder picker browses the machine running HistoPilot, not the computer running the browser. To train models, also create the training environment; see [deployment](docs/deployment.md#training-environment). For TRIDENT feature extraction, see [deployment](docs/deployment.md#trident-feature-extraction).

## The workflow

1. **Datasets:** import a CSV/XLSX slide table, map slide and patient identifiers, link slide files and freeze a dataset version.
2. **Slide features:** extract patch features with TRIDENT, or attach existing ones, then validate them and freeze a feature bundle.
3. **Targets & splits:** choose the prediction target and freeze fixed training and testing sets. The testing set becomes a test cohort. Steps 2 and 3 can run in either order.
4. **Experimental Setup:** combine a dataset, a target/split version and a feature bundle; design the cross-validation folds, model recipes and predictor choices; then freeze the setup.
5. **Experiments:** start a frozen setup, follow its runs, and review cross-validated results and ready predictors.
6. **Evaluate models** or **Run inference:** score ready predictors on a labeled test cohort, or apply them to unlabeled slides for predictions, review and export.
7. **Clinical utility** and **Model interpretation:** examine calibration, thresholds and net benefit, and review attention maps and top patches on the slides.

Compute runs through the **Task Center**, one queue per machine shared by all projects. Models include ABMIL, nnMIL, mean and max pooling MIL, and linear and MLP probes on slide embeddings.

## A look inside

The **BLCA demo** follows 138 synthetic slides: 62 for development and 76 for testing. All records, scores and curves are illustrative. It includes no real slide images or attention maps.

| Prepare data | Follow training |
| :---: | :---: |
| [![Synthetic BLCA dataset and cohort counts](docs/assets/blca/dataset.png)](docs/assets/blca/dataset.png) | [![Synthetic BLCA training and validation curves](docs/assets/blca/training.png)](docs/assets/blca/training.png) |

| Evaluate predictions | Explore clinical utility |
| :---: | :---: |
| [![Synthetic BLCA ROC curve and confusion matrix](docs/assets/blca/evaluation.png)](docs/assets/blca/evaluation.png) | [![Synthetic BLCA decision curve and threshold tradeoffs](docs/assets/blca/clinical-utility.png)](docs/assets/blca/clinical-utility.png) |

## Documentation

| Guide | For |
| --- | --- |
| [User guide](docs/user-guide.md) | Running a study end to end, stage by stage |
| [Methods](docs/methods.md) | How splits, cross-validation, metrics, intervals and predictors are defined |
| [Deployment](docs/deployment.md) | Installation, runtime environments, configuration and remote access |
| [BLCA demo](docs/blca-demo.md) | The synthetic walkthrough |
| [Task Center](docs/task-center.md) | How compute is queued, admitted, cancelled and recovered |
| [Architecture](docs/architecture.md) | Runtime, storage, execution model and code map |
| [API](docs/api.md) | The local HTTP API |
| [Contributing](CONTRIBUTING.md) | Development environments, tests and conventions |

<sub>No project license has been selected. Third-party models and backends retain their own licenses.</sub>
