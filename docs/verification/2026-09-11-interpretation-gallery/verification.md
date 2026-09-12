# Interpretation gallery verification — 2026-09-11

The redesigned module follows one sequence: choose a frozen ABMIL predictor, a shared feature bundle and representation, open a slide folder, then click a slide or select a batch. Changes are in the local `main` working tree. Development did not start or restart HistoPilot.

## UI verdict

The previous per-slide path form was a poor fit for an extracted cohort. The gallery now keeps the shared input choices together and makes the slide image the primary action. Search covers the folder before pagination. Checkboxes are distinct from the single-slide action, and the selected count explicitly includes other pages and searches. Missing or ambiguous feature matches remain visible with a reason.

Completed batches show multiple actual attention overlays, with individual progress, errors, reuse notices and access to the detailed viewer. Predictor, bundle, representation, folder and evidence-link changes discard the current selection and results. Earlier saved studies retain their own provenance. Network failures retain the exact operation and request for a retry, preventing accidental resubmission with different inputs.

The browser checks used the real React page and components compiled into a standalone local HTML fixture. API responses and job transitions were controlled test doubles; image and attention payloads came from a tiny genuine ABMIL fixture. This verifies UI behavior independently of a running application server. It is not a vendor-WSI performance benchmark.

Verified in the browser:

- Lazy thumbnails, 24-slide pagination, unavailable-slide controls and case-insensitive search.
- Empty search results, delayed-response races and selection persistence across searches and pages.
- Single ensemble attention with mean and both member options; single refit attention without stale ensemble controls.
- Exact batch membership and multiple completed overlay cards.
- Lost accepted response followed by the same operation retry without another job.
- Queued batch results becoming completed overlays, including the gallery status labels.
- Predictor, bundle, pack and folder changes resetting the appropriate context.
- Desktop width 1,440 and mobile width 390 without horizontal overflow; no runtime browser errors or unexpected network requests.

Screenshots: [desktop controls](gallery-desktop.png), [mobile gallery](gallery-mobile.png), [single ensemble](ensemble-single.png), [batch overlays](ensemble-batch.png), [single refit](refit-single.png).

The final representation controls also disable incompatible dtypes and automatically choose a compatible predictor pack when appropriate. Explicit user choices survive inventory refreshes. FP16 predictor / FP32 original-feature cases have dedicated regression coverage.

## Backend verdict

The bundle inventory supplies exact feature IDs, imported aliases and coordinate paths. Duplicate names and aliases are rejected rather than matched approximately. Folder scans are bounded, recursive, deterministic and reject unsafe path traversal. Authentication also covers thumbnails and attention/image artifacts.

Packed attention reads `features.bin` and `coords.bin` using the selected slide's indexed offset and row count. It checks encoder, dimensions, dtype, geometry, tensor hashes and file stamps. Native packs and verified legacy OceanPath packs are supported. Original HDF5 containers are unnecessary for packed worker execution; the bundle registry's existing source-freshness policy still applies during selection. A refit uses its frozen checkpoint; an ensemble uses every frozen checkpoint and averages normalized attention and probabilities. Complete bags are retained.

Each selected slide has its own durable study and job. Receipts preserve launch intent before queueing, so a lost response or quickly failed job cannot silently advance to a new attempt on the same operation. A new request can explicitly retry. Preparation does not hold the project lifecycle lock, allowing workers and cleanup/status operations to acquire it. Batch preparation is bounded to 45 seconds, and accepted submissions survive a limit or an individual input failure. Repeated folder scans are avoided within a batch while selected tensors are revalidated.

Saved studies preserve the predictor, bundle, feature/pack dependencies and optional evaluation/clinical evidence links. Cleanup checks include an external-slide bundle shared by an active or archived attention study. Old manually prepared HDF5 studies remain launchable and resumable.

## Genuine inference evidence

The retained fixture at `/tmp/histopilot-interpretation-gallery/real-fixture` contains two nested slide folders, original HDF5 files, a shared native pack, and separate native/packed refit/ensemble plans and outputs. The slides have seven and five patches. Pack order is deliberately reversed relative to selection order, testing both a nonzero offset and different patch counts. The ensemble uses two distinct checkpoints.

Native and packed probabilities, coordinates and attention outputs agree for both methods. A frozen worker archive also executed the two-slide packed ensemble. Storage tests cover tampering, mismatched evidence, geometry, legacy pack proof and source-independent packed loading.

## Validation results

| Check | Result |
| --- | --- |
| `npm test -- --reporter=dot` in `web` | 314 passed across 47 files |
| `tests/test_interpretation_gallery.py` | 20 passed |
| `tests/test_interpretation.py` | 28 passed |
| `tests/test_attention_packs.py tests/test_packed_storage.py tests/test_pack_import.py` | 76 passed |
| `tests/test_attention_execution.py tests/test_inference_execution.py` in `.venv-training` | 18 passed with real CPU inference |
| `tests/test_workspace_cleanup.py tests/test_compute_archive.py tests/test_compute_jobs.py` | 32 passed |
| Ruff, TypeScript and production frontend build | Passed |
| [Browser assertions](browser-checks.json) | 26 checks passed |
| Local wheel build and package-content verification | Passed |

Python suites used `.venv/bin/pytest -q` unless explicitly marked `.venv-training/bin/python -m pytest -q`. In-process API tests ran with local socket permission; they did not start a HistoPilot server. The production build retains the existing large-JavaScript-chunk advisory. The package contains the current gallery services, packed reader, frozen-worker snapshot inputs and rebuilt frontend assets. Importing the control API does not load Torch, Lightning, HDF5, Pillow or OpenSlide.

## Practical limits

Batch overview cards display up to 5,000 patches with an explicit partial-preview note. Detailed viewers display up to 100,000 patches per viewport; zooming provides complete smaller-region inspection, and exports retain all patches. Folder scans stop with an explanation when exceeding 20,000 entries, 10,000 slides, 24 nested levels or five seconds. A visualization batch accepts at most 128 slides.

Attention remains class-independent ABMIL pooling attention, rather than a class-specific probability map. The tiny CPU fixtures establish numerical and orchestration correctness; production-scale WSI storage throughput, GPU capacity and clinical validation require representative deployment data.
