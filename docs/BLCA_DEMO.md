# BLCA demo

The BLCA demo explains HistoPilot's bladder workflow using synthetic records and results. Open **BLCA demo** from the start page, or visit [`?project=blca-demo-v1#overview`](http://127.0.0.1:8787/?project=blca-demo-v1#overview) on a locally running service. Use your configured port if different.

It follows the current Bladder project's aggregate cohort structure, target definitions, feature representation and model-development sequence. All row identifiers, row-level labels, predictions, loss histories and resource histories are newly generated for the example. The original Bladder project and its jobs are unchanged.

## What the example contains

| Population | Slides | WHO 2022 low | WHO 2022 high |
| --- | ---: | ---: | ---: |
| Development | 62 | 33 | 29 |
| Grade-2 test cohort | 76 | 54 | 22 |
| Total | 138 | 87 | 51 |

These are aggregate reference counts. Synthetic rows are created to match them; they are not renamed or shuffled copies of real clinical rows. The grade-2 test cohort uses **WHO 1973 = 2**, while the prediction target is **WHO 2022 low/high**, with high as the positive class. The two grading fields serve different roles.

Patient identity is unverified in the reference workflow: its slide/case identifier is not evidence of a patient identifier. The demo therefore uses **slide fallback groups** and slide-level evaluation. Its 138 slide records must not be described as 138 independent patients, and the grouping does not establish patient-disjoint development and test populations.

## Walk through the pipeline

Each module opens a Stage 0 record library. Search or filter the examples, open a record, then use Back and Next to review one step at a time. The demo is read-only: it explains completed work without creating projects, changing records or launching compute jobs.

| Module | What to inspect | What it explains |
| --- | --- | --- |
| Datasets | Synthetic slide records, grade fields and identity notes | Importing a source table, reviewing mappings and freezing the dataset |
| Targets & splits | Development membership, binary target and five folds | Separating target labels, cohort eligibility, fitting/validation roles and development assessment |
| Slide features | UNI metadata with 1,024 feature dimensions and bundle evidence | Attaching or extracting representations, validating coverage and freezing a feature bundle |
| Experiments | Input references, ABMIL recipes, runs and predictor choices | Following one frozen configuration through training, checkpoints and predictor creation |
| Test cohorts | The separate 76-slide WHO 1973 grade-2 population | Freezing an evaluation population independently of model selection |
| Evaluate models | Synthetic predictions and metrics for the baseline v2 P75 refit | Applying a ready predictor to the separate test cohort and inspecting errors |
| Clinical utility | Synthetic calibration, operating-point and utility summaries | Looking beyond discrimination to confidence, false positives, false negatives and threshold tradeoffs |
| Model interpretation | An explanatory schematic | Where slide attention belongs in the pipeline and what additional inputs it requires |

### Development and predictor creation

The recipe follows the reference configuration: UNI patch features, gated ABMIL, five development folds, training and split seed 42, a maximum of 40 epochs and a training bag limit of 4,096 patches. These describe the illustrative recipe; the demo does not contain feature tensors or execute training.

Runs appear before predictor creation. Select a run to inspect synthetic loss history, checkpoint metadata, diagnostics and artifact references. Resource history sits below the runs and illustrates CPU, GPU, RAM and VRAM monitoring. Those curves are generated examples, not measurements of the machine hosting the demo.

The baseline v2 example includes a fold ensemble and a P75 refit; baseline v3 illustrates a P50 refit. A fold ensemble combines the saved fold predictors. Refit trains one predictor on the development population using a budget derived from the best fold epochs: P50 is the median and P75 is the 75th percentile. The demo's chosen epochs and derived refit budgets are invented. In a real project these operations require verified checkpoints and a compatible training runtime; the demo only explains their records and relationships.

### Reading the results

Synthetic development predictions illustrate very strong development discrimination. The separate grade-2 test example is strong but imperfect, with false positives and false negatives that make calibration and threshold review meaningful. This qualitative pattern follows the reference workflow, but every displayed probability and derived metric is generated for the demo. No displayed score is a report of real Bladder performance.

Development OOF predictions assess held-out development folds. They can support configuration comparison, but choosing a configuration from those scores does not produce an independent final-test estimate. Keep that distinction when moving to the test-cohort example. Calibration and clinical-utility plots explain how to examine predictions; they do not establish a clinically validated decision rule.

## Synthetic data boundary

The distributed demo includes generated metadata and illustrative numerical results. It excludes real clinical rows, patient/case/slide identifiers, source filesystem paths, whole-slide images, image crops, extracted embeddings, model weights, trained checkpoints and real run logs. Artifact references describe the example and do not provide access to local research files.

Model interpretation is an explanatory view only. There are no patient slide pixels or computed model-attention maps in this demo. Actual interpretation in a local project uses compatible slides, validated feature coordinates and verified ABMIL predictors.

The BLCA view does not expose record mutation, training, filesystem browsing or local system controls. Available exports contain synthetic records and remain labeled as examples. Opening it does not import, modify or publish the reference Bladder project. Earlier CRC demo links using `synthetic-v1` remain compatible; **Open BLCA demo** is the default example entry point.

## Reproduce the fixture

The deterministic generator is [`histopilot/application/blca_demo.py`](../histopilot/application/blca_demo.py), using seed `20260912`. It encodes the aggregate structure above and creates synthetic rows, split memberships, probabilities, epoch histories and resource samples. Metrics are calculated from the released synthetic prediction rows. The generator does not read the original Bladder project or any research data.

Check the packaged fixture against a fresh generation:

```bash
uv run python -m histopilot.application.blca_demo --check
```

After intentionally changing the generator, regenerate its source-controlled JSON:

```bash
uv run python -m histopilot.application.blca_demo --write
```

With no arguments, the command prints the generated workspace JSON. The wheel packages [`histopilot/resources/blca_demo_workspace.json`](../histopilot/resources/blca_demo_workspace.json) through the existing `resources/*.json` package-data rule. Only this self-contained synthetic resource is needed to load the example; no export of the real project is used.

## Run it locally

From the repository root:

```bash
uv sync --locked
npm --prefix web ci
npm --prefix web run build
uv run python scripts/bundle_web.py
```

Then start the service manually:

```bash
uv run histopilot serve --no-browser
```

Open `http://127.0.0.1:8787` and choose **Open BLCA demo**. No research-data roots, downloaded weights, training environment or GPU are required. Build and verification commands do not start or restart the server. See [deployment](deployment.md) for Vite development, custom ports, packaging and SSH forwarding.
