# OceanPath integration plan

OceanPath is a candidate execution backend and source of implementation patterns. HistoPilot owns dataset identity, cohorts, patient grouping, experiment specifications, and provenance. A future adapter will translate those contracts into an explicit execution plan run by an isolated Python worker. FastAPI must never load OceanPath models or CUDA state.

This plan comes from a **read-only inspection of the adjacent local OceanPath checkout** at commit `384e8a3ffefe148f7b006effefbc71eb97020ce1`, including its `README.md`, `pyproject.toml`, and the source paths below. Paths are relative to OceanPath. They describe that checkout, not a pinned HistoPilot dependency. Nothing from OceanPath is imported or executed by this skeleton.

## Concrete candidates

| HistoPilot boundary | Observed OceanPath source | Proposed future use |
| --- | --- | --- |
| PFM adapter | `src/oceanpath/data/feature_extract.py`: `TridentExtractionConfig`, `run_pipeline`, `compute_encoder_fingerprint`, output validation | Translate a HistoPilot extraction request into explicit TRIDENT settings; return feature artifact references and extraction provenance. |
| Feature storage adapter | `src/oceanpath/data/mmap_builder.py`: `MmapBuildConfig`, `build_mmap`, `load_mmap_index`, `validate_mmap_dir` | Evaluate optional worker-side mmap materialization after the initial HDF5 feature adapter; retain HistoPilot slide IDs, coordinates, dimensions, and source fingerprints. |
| Split integration | `src/oceanpath/data/splits.py`: `SplitConfig`, `generate_splits`, `verify_split_integrity` | Evaluate patient-grouped split generation and integrity checks. HistoPilot should persist an explicit frozen assignment before a trainer consumes it. |
| MIL adapter | `src/oceanpath/models/base.py`, `models/abmil.py`, `models/__init__.py`; `src/oceanpath/modules/train_module.py` | Start with ABMIL and translate the `BaseMIL`/`MILOutput` tensor contract and training outputs into HistoPilot runs, checkpoint artifacts, predictions, and attention references. |
| Evaluation application service | `src/oceanpath/eval/core.py`: `compute_metrics`, `aggregate_to_patient_level`, `bootstrap_ci`; `eval/comparison.py` | Adapt patient-level evaluation and comparison after defining label encoding, prediction schema, aggregation policy, and held-out evaluation rules. |
| Explorer artifact preparation | `src/oceanpath/eval/attention.py`: `load_h5_coords`, `build_attention_overlay`, `visualize_slide_attention` | Prepare attention artifacts with coordinate and run metadata. A dedicated WSI port must still own image-reader and region access. |
| Job execution | `src/oceanpath/pipeline/dag.py`: `Stage`, `PipelineRunner`; `pipeline/transactions.py`: `atomic_output`, transaction sidecars | Evaluate stage fingerprints, output validation, and atomic publication behind the job port. Persist HistoPilot status transitions independently of worker process lifetime. |
| Provenance | `src/oceanpath/utils/repro.py`: `config_fingerprint`, `manifest_hash`, `capture_provenance`, `save_provenance` | Translate config/environment records and checksums into a versioned HistoPilot provenance record linked to immutable inputs and output artifacts. |

These are candidates for review, not claims that the interfaces already match. In particular, OceanPath's README describes some modules absent from the inspected `src/oceanpath` tree; the table uses observed files only.

## Integration shape

Prefer a separately installed, version-pinned backend over vendoring a research repository wholesale. Keep OceanPath imports inside an isolated worker environment so starting the control service, opening the UI, or importing HistoPilot domain types never imports PyTorch, Lightning, Hydra, or native image-reader dependencies. The adapter may generate a safe argument list for a separately installed backend CLI; pin and verify the backend version and command contract before implementing that route.

The inspected OceanPath package requires Python 3.10+ and includes a broad scientific/GPU dependency set, including PyTorch, Lightning, Hydra, TRIDENT, and native extensions. HistoPilot targets Python 3.11+ with a lightweight control-service dependency set: FastAPI, Pydantic, SQLAlchemy, Uvicorn, and CLI/configuration support. Test a dedicated backend environment before defining an optional compute dependency group; do not copy OceanPath's GPU/scientific dependency list into the control-service package.

An adapter should perform the following steps:

1. Resolve frozen HistoPilot IDs to a read-only input manifest and artifact locations.
2. Translate explicit configuration, including patient grouping, label encoding, feature dimension, coordinate units, seed, encoder identity, and checkpoint references.
3. Write backend inputs to a run-specific working directory, build an `ExecutionPlan`, submit through an isolated-process executor, and capture the exact argument list and environment. Keep shell evaluation out of backend argument construction.
4. Convert backend progress and failure information to HistoPilot job/run states without declaring completion from process launch alone.
5. Validate outputs, calculate checksums, and publish artifacts atomically before linking a completed `Result` to the originating `Run`.

Do not infer patient identity from a filename inside the training adapter. Resolve patient → specimen → slide centrally during dataset construction. Likewise, do not regenerate a train/test split inside a trainer or silently change an encoder configuration when locating cached features.

## License status

No `LICENSE`, `COPYING`, or `NOTICE` file was found in the inspected checkout, and neither its README nor its package metadata declared a license. This records the observed repository state; it does not establish permission to redistribute its code. Confirm and document the intended license before copying or vendoring any OceanPath implementation. This skeleton copies no OceanPath code, weights, or data.

TRIDENT, individual model checkpoints, and other optional backends require their own dependency/version and license review when integrated. No model weights are downloaded by this repository. The current API model registry describes planned choices and does not claim backend or checkpoint availability.

## First integration acceptance criteria

Use one explicit fixture-scale dataset, one patient-grouped split, one encoder, and ABMIL. Require an end-to-end test that verifies:

- Each output prediction resolves to the correct patient and slide identities, and patients never cross partitions.
- Feature rows remain aligned with patch coordinates and the same encoder/preprocessing fingerprint.
- A failure leaves no result marked complete; a resumed job preserves the original inputs and run identity.
- The metric and an attention region resolve to a complete run manifest, with real artifact paths and checksums.
- Starting FastAPI, using the React UI, and inspecting metadata still work without the OceanPath environment or CUDA initialization.
- The same validated `ExperimentSpec` is accepted through the GUI/API and CLI, and progress is relayed from worker records rather than invented in the browser.
