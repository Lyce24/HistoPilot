# Synthetic CRC / KRAS example

These files describe the fictional workspace used by the browser UI: **24 patients, 28 specimens, and 28 slides**. The first four patients each have two specimens/slides; every other patient has one. They contain no real patient data, slide pixels, embeddings, trained models, or clinical evidence.

| File | Contents |
| --- | --- |
| `clinical.csv` | Fictional patient IDs and three invented collection sites. |
| `labels.csv` | Illustrative MSI, BRAF, and KRAS values keyed by patient ID. |
| `slides.csv` | Explicit patient → specimen → slide mapping, specimen type, placeholder WSI URI, and demo review state. |
| `manifest.json` | Illustrative provenance for `experiment-demo-01` / `run-demo-01`, including patient partition assignments and result lineage. |
| `experiment-spec.json` | A schema-valid experiment input shared by GUI/API and CLI; execution is not yet implemented. |

The same synthetic records seed the service through [`histopilot/resources/demo_workspace.json`](../../histopilot/resources/demo_workspace.json). The browser reads workspace and model data from the API. Editing these example CSV/JSON files does not import data or alter saved records in an existing workspace. Keep the packaged seed and the examples aligned when changing the demonstration data.

`manifest.json` uses `schema_version: "0.1.0"`, `mode: "synthetic-demo"`, and `executable: false`. It is an illustrative export format, **not** an executable experiment configuration, a Python dataclass serialization format, or a supported import schema. The provenance export uses this format for an example result. Workspace export includes server-owned saved state as a separate format. Cohort snapshots and experiment drafts are now persisted in SQLite rather than held only in browser memory.

All `demo://` locations are placeholders. They do not resolve to files, model weights, feature stores, or checkpoints; hashes and environment fields marked `placeholder:` are intentionally uncomputed. Names such as `clinical.csv` and `labels.csv` illustrate the future source-table relationship, not a working URI resolver.

The synthetic split assigns patients 001–016 to training, 017–020 to validation, and 021–024 to test. It is a fixed illustration tagged with seed 42, not an implementation of seeded or stratified split generation. Example performance scores are invented and are not calculated from the CSV labels or fitted models. An attention display is a synthetic illustration, not a clinical interpretation.

Validate the separate experiment input with:

```bash
histopilot run examples/crc_kras/experiment-spec.json --validate-only
```

See [`examples/experiment.schema.json`](../experiment.schema.json) for the canonical schema. Passing schema validation does not verify real source paths, artifacts, model availability, or scientific readiness. Running without `--validate-only` fails explicitly until worker execution is implemented.

When implementing the first real workflow, replace the placeholder sources through the dataset importer, validate patient identity and labels, freeze a real dataset/cohort/split, and create a new run with actual output artifacts. Preserve these small fixtures as the clearly marked demo path.
