# Interpretation stages verification — 2026-09-11

Implemented in the local `main` checkout. Work remains uncommitted. The HistoPilot server was neither started nor restarted.

## Interface verdict

Interpretation now has separate selection, preparation, selected-results and focused-slide pages. The model and frozen shared features determine the dataset slide folder. Thumbnail clicks only select slides; Continue explicitly submits the batch. Search and selection survive Back navigation. Every selected job is polled independently of whether its card is visible, and results require verified completion for the entire selected batch.

The focused page renders one slide viewer. At a 1440 × 1080 viewport with the real application sidebar and content padding, the canvas measured approximately 868 × 756 pixels in a 1130-pixel content region (77% of available width). A 250-pixel rail contains the selected original crop and scrollable ranked patch cards. Hide patches and Expand view offer more room. Container-width breakpoints put the rail below the slide on narrower screens.

The initial fit uses all extracted patch footprints, including low-attention patches, rather than the full slide's blank margin. A delayed coverage response never overrides manual pan/zoom. Fit slide remains available. Numbered top-10/top-20 boxes, ensemble-member changes and original crop selection remain linked. A browser check caught implicit CSS grid sizing that clipped the compact crop; explicit constrained grid tracks now preserve the complete image on desktop and mobile.

## Backend verdict

Dataset-folder inference uses the selected feature bundle's own frozen dataset, including external cohorts. Saved import roots support nested folders. A legacy fallback requires one direct recorded parent; ambiguous or inaccessible locations return a repair message directing the user to Datasets. Path containment and source identity checks remain enforced.

Completed and active attention can be reused when only execution resource reservations change. Reuse still requires matching scientific inputs: predictor, bundle, pack, slide geometry, source/tensor identity and evidence links. Corrupt result receipts are rejected. Exact operation replay retains accepted jobs across lost responses; failed-slide retries retain immutable geometry and lineage.

Top-attention responses provide the clipped level-0 bounding box of all coordinate footprints. Array-backed maps calculate bounds during the existing bounded global-ranking scan; legacy maps use the authenticated compact cache. Coverage tests include chunk boundaries, fractional footprints, low-attention extrema, edge clipping and cache isolation.

## Final checks

- Frontend: **337 tests across 51 files passed**, including stage gating, saved-study source hydration, URL/session restoration and exact retry persistence. TypeScript and production Vite build passed.
- Backend: **97 focused tests passed**: 58 attention/ranking/authenticated-crop/manual-study cases and 39 dataset-folder/gallery/resource-reuse cases. Three architecture checks passed; base API import avoided training and imaging runtimes. Ruff check, formatting and Git whitespace checks passed.
- Browser: **52 checks passed** against the actual React components, with zero browser errors or unexpected network requests. Coverage includes search, packed features, 10-slide batch progression, queued/running/completed reuse, offscreen polling, partial completion, mixed failures, safe retry after reload, refit/ensemble/member rendering, saved-study deep links, browser Back/Forward, exact source restoration, top-20 patch selection, desktop/mobile sizing, complete crop rendering, expand/Escape, delayed coverage and verified-error recovery.
- Packaging: rebuilt `dist/histopilot-0.1.0.dev0-py3-none-any.whl` and verified the wheel's four frontend files exactly match `web/dist` and `histopilot/static`; changed backend files also match byte for byte. The final JavaScript asset is `index-edo_TMAq.js`, with `index-CvVYpzQm.css`.

## Verification method and limits

Browser verification used a static inline build of the actual React interpretation components and application shell CSS. There was no development server. Its APIs were deterministic in-browser fixtures with persistent mock job records for reload testing. Attention weights came from the earlier real CPU ABMIL fixture with two distinct checkpoints and 30 patches. The 116,736 × 101,376 large-slide case used synthetic image/coordinate placement with the real fixture weights; it did not evaluate a patient WSI. Browser workflows complement the in-process backend/API tests rather than constituting a live production deployment test.

The base service import remains isolated from Torch, Lightning, HDF5, Pillow and OpenSlide. The production build retains the existing large-chunk advisory; it is not a build failure.

See [selection page](selection.png), [desktop viewer](focused-desktop.png), [expanded viewer](focused-expanded.png), [mobile viewer](focused-mobile.png), [computation](computation.png), [selected results](results.png), and [individual browser checks](browser-checks.json).
